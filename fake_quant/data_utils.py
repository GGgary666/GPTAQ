import os
import datasets
import random
import transformers


def _load_tokenizer(model, hf_token=None):
    kwargs = {'use_fast': False}
    if hf_token:
        kwargs['token'] = hf_token
    return transformers.AutoTokenizer.from_pretrained(model, **kwargs)


def _load_wikitext_split(split, dataset_dir=None):
    """Load WikiText-2 raw from a local directory when possible.

    Accepts either a ``datasets.save_to_disk`` folder or parquet files whose
    names contain the split. Falls back to the Hugging Face hub.
    """
    candidates = []
    if dataset_dir:
        candidates.extend([
            os.path.join(dataset_dir, 'wikitext-2-raw-v1'),
            dataset_dir,
        ])
    for path in candidates:
        if not os.path.isdir(path):
            continue
        dict_file = os.path.join(path, 'dataset_dict.json')
        split_dir = os.path.join(path, split)
        if os.path.exists(dict_file):
            return datasets.load_from_disk(path)[split]
        if os.path.isdir(split_dir):
            return datasets.load_from_disk(split_dir)
        parquets = []
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith('.parquet') and split in fname:
                    parquets.append(os.path.join(root, fname))
        if parquets:
            return datasets.load_dataset('parquet', data_files=sorted(parquets), split='train')
    try:
        return datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)
    except Exception:
        return datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False, dataset_dir=None):
    tokenizer = _load_tokenizer(model, hf_token)

    if eval_mode:
        testdata = _load_wikitext_split('test', dataset_dir)
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc

    traindata = _load_wikitext_split('train', dataset_dir)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False, dataset_dir=None):

    tokenizer = _load_tokenizer(model, hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader



def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False, dataset_dir=None):
    tokenizer = _load_tokenizer(model, hf_token)

    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False,
    dataset_dir=None,
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode, dataset_dir)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode, dataset_dir)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode, dataset_dir)
