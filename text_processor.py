import torch
from transformers import DistilBertTokenizer, DistilBertModel

def text_processor(sentence, tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-cased'), max_length=256, add_special_tokens=True, truncation=True):
    tokenizer = tokenizer
    text = str(sentence)
    embeddings = tokenizer(text,
                           add_special_tokens = add_special_tokens,
                           max_length = max_length,
                           truncation = truncation,
                           padding = 'max_length')


    input_ids = torch.unsqueeze(torch.tensor(embeddings['input_ids']), dim=0)
    attention_mask = torch.unsqueeze(torch.tensor(embeddings['attention_mask']), dim=0)

    return input_ids, attention_mask


if __name__ == "__main__":
    input_ids, attention_mask = text_processor('diy-tools-materials,x4 tins farrow ball estate emulsion no. 75 ball green unopened discounts for multiples')
    print(input_ids.size(), attention_mask.size())
    model = DistilBertModel.from_pretrained('distilbert-base-cased')
    embedding = model(input_ids, attention_mask)
    print(embedding)
