from mmdet.registry import MODELS
from torch import nn
import os
from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer
import torch
from transformers import CLIPTextModel
from .transformer import Transformer
from .prompt_engineering import prompt_engineering, get_prompt_templates


def build_lang_encoder(config_encoder, tokenizer, verbose, **kwargs):
    model_name = config_encoder['NAME']

    if model_name == 'PathClipTextEncoder' or model_name == "ClipOriTextEncoder":
        return CLIPTextModel.from_pretrained(config_encoder['PRETRAINED_TOKENIZER'])
        
    return Transformer(
        context_length=config_encoder['CONTEXT_LENGTH'],
        vocab_size=tokenizer.vocab_size,
        width=config_encoder['WIDTH'],
        layers=config_encoder['LAYERS'],
        heads=config_encoder['HEADS'],
        autogressive=config_encoder.get('AUTOGRESSIVE', True))

def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', '/YOUR_PATH/clip-vit-base-patch32/'
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer, from_slow=True)
    elif config_encoder['TOKENIZER'] == 'path-clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', '/YOUR_PATH/clip-vit-base-patch32/'
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'])

    return tokenizer


@MODELS.register_module()
class LanguageEncoder(nn.Module):
    def __init__(
        self,
        tokenizer_type,
        verbose,
        lang_encoder,
        dim_lang,
        dim_projection,
        max_token_num,
        load_from="",
        queue_operator={},
        freeze=True,
        seem=True,
    ):
        super().__init__()
        self.tokenizer = build_tokenizer(lang_encoder)
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = build_lang_encoder(lang_encoder, self.tokenizer, verbose=verbose)
        self.max_token_num = max_token_num
        self.lang_proj = nn.Parameter(torch.empty(dim_lang, dim_projection))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.load_from = load_from
        self.seem = seem

        if load_from:
            self._init_weights()
        
        if freeze:
            self._freeze()
        
    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.lang_encoder.parameters():
            param.requires_grad = False
        
    def _init_weights(self):
        if self.load_from:
            self.load_state_dict(torch.load(self.load_from), strict=False)
    
    def get_text_embeddings(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, store_buffer=None):
        # import pdb; pdb.set_trace()
        if not is_eval:
            if prompt:
                # randomly sample one template
                arbitary_concepts = [
                    prompt_engineering(class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
                if add_bgd:
                    arbitary_concepts.append("A background in coco.")
            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    # import pdb; pdb.set_trace()
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                templates = get_prompt_templates()
                clss_embeddings = []
                if prompt:
                    for clss in class_names:
                        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
                        clss_embeddings.append(extract_mean_emb(txts))
                else:
                    for clss in class_names:
                        clss_embeddings.append(extract_mean_emb([clss]))

                if add_bgd:
                    txts = ["A background in coco."]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def reset_text_embeddings(self, name='default'):
        pass

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret

    def forward_language(self, texts, norm=True):
        # import pdb; pdb.set_trace()
        # from mmengine.analysis import get_model_complexity_info
        # flops = get_model_complexity_info(self.lang_encoder, None, (texts[0], texts[1]))

        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip' or self.tokenizer_type == 'path-clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]
        if self.seem:
            x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def forward_language_token(self, texts, norm=False):
        x = self.lang_encoder(*texts)
        token_x = x['last_hidden_state']

        if self.tokenizer_type == 'clip' or self.tokenizer_type == 'path-clip':
            class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]
        else:
            class_x = token_x[:, 0]

        if self.seem:
            class_x = class_x @ self.lang_proj
            token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x
    
    def compute_similarity(self, v_emb, name='default', fake=False):
        if fake:
            return None
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output

if __name__ == '__main__':
    tokenizer_type = "clip"
    verbose = True
    lang_encoder = {'NAME': 'ClipTextEncoder', "TOKENIZER": "clip", 'CONTEXT_LENGTH': 77, 'WIDTH': 512, 'HEADS': 8, 'LAYERS': 12, 'AUTOGRESSIVE': True}
    dim_lang = 512
    dim_projection = 512
    max_token_num = 77
    
    model_custom = LanguageEncoder(tokenizer_type, verbose, lang_encoder, dim_lang, dim_projection, max_token_num, 
                                   load_from=os.path.join("/YOUR_PATH/pretrain_check/lang_encoder.pt"))