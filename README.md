# spacybert: Bert inference for spaCy
[spaCy v2.0](https://spacy.io/usage/v2) extension and pipeline component for loading BERT sentence / document embedding meta data to `Doc`, `Span` and `Token` objects. The Bert backend itself is supported by the [Hugging Face transformers](https://github.com/huggingface/transformers) library.

## Installation
`spacybert` requires `spacy` v2.0.0 or higher.

## Usage
### Getting BERT embeddings for single language dataset
```
import spacy
from spacybert import BertInference
nlp = spacy.load('en')
```

Then either use BertInference as part of a pipeline,
```
bert = BertInference(
    from_pretrained='path/to/pretrained_bert_weights_dir',
    set_extension=False)
nlp.add_pipe(bert, last=True)
```
Or not...
```
bert = BertInference(
    from_pretrained='path/to/pretrained_bert_weights_dir',
    set_extension=True)
```
The difference is that when `set_extension=True`, `bert_repr` is set as a property extension for the Doc, Span and Token spacy objects. If `set_extension=False`, the `bert_repr` is set as an attribute extension with a default value (`=None`). The attribute computes the correct value when `doc._.bert_repr` is called.

Get the Bert representation / embedding.
```
doc = nlp("This is a test")
print(doc._.bert_repr)  # <-- torch.Tensor
```

### Getting BERT embeddings for multiple languages dataset.
```
import spacy
from spacy_langdetect import LanguageDetector
from spacybert import MultiLangBertInference

nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
bert = MultiLangBertInference(
    from_pretrained={
        'en': 'path/to/en_pretrained_bert_weights_dir',
        'nl': 'path/to/nl_pretrained_bert_weights_dir'
    },
    set_extension=False)
nlp.add_pipe(bert, after='language_detector')

texts = [
    "This is a test",  # English
    "Dit is een test"  # Dutch
]
for doc in nlp.pipe(texts):
    print(doc._.bert_repr)  # <-- torch.Tensor
```
When language_detector detects languages other than the ones for which pre-trained weights is specified, by default `doc._.bert_repr = None`.

## Available attributes
The extension sets attributes on the `Doc`, `Span` and `Token`. You can change the attribute name on initializing the extension.
| | | |
|-|-|-|
| `Doc._.bert_repr` | `torch.Tensor` | Document BERT embedding |
| `Span._.bert_repr` | `torch.Tensor` | Span BERT embedding |
| `Token._.bert_repr` | `torch.Tensor` | Token BERT embedding |
| | | |

## Settings
On initialization of `BertInference`, you can define the following:

| name | type | default | description |
|-|-|-|-|
| `from_pretrained` | `str` | `None` | Path to Bert model directory or name of HuggingFace transformers pre-trained Bert weights, e.g., `bert-base-uncased` |
| `attr_name` | `str` | `'bert_repr'` | Name of the BERT embedding attribute to set to the `._` property |
| `max_seq_len` | `int` | 512 | Max sequence length for input to Bert |
| `pooling_strategy` | `str` | `'REDUCE_MEAN'` | Strategy to generate single sentence embedding from multiple word embeddings. See below for the various pooling strategies available. |
| `set_extension` | `bool` | `True` | If `True`, then `'bert_repr'` is set as a property extension for the `Doc`, `Span` and `Token` spacy objects. If `False`, the `'bert_repr'` is set as an attribute extension with a default value (`None`) which gets filled correctly when called in a pipeline. Set it to `False` if you want to use this extension in a spacy pipeline. |
| `force_extension` | `bool` | `True` | A boolean value to create the same 'Extension Attribute' upon being executed again |

On initialization of `MultiLangBertInference`, you can define the following:

| name | type | default | description |
|-|-|-|-|
| `from_pretrained` | `Dict[LANG_ISO_639_1, str]` | `None` | Mapping between two-letter language codes to path to model directory or HuggingFace transformers pre-trained Bert weights |
| `attr_name` | `str` | `'bert_repr'` | Same as in BertInference |
| `max_seq_len` | `int` | 512 | Same as in BertInference |
| `pooling_strategy` | `str` | `'REDUCE_MEAN'` | Same as in BertInference |
| `set_extension` | `bool` | `True` | Same as in BertInference |
| `force_extension` | `bool` | `True` | Same as in BertInference |

## Pooling strategies
| strategy | description |
|-|-|
| `REDUCE_MEAN` | Element-wise average the word embeddings |
| `REDUCE_MAX` | Element-wise maximum of the word embeddings |
| `REDUCE_MEAN_MAX` | Apply both `'REDUCE_MEAN'` and `'REDUCE_MAX'` and concatenate. So if the original word embedding is of dimensions `(768,)`, then the output will have shape `(1536,)` |
| `CLS_TOKEN`, `FIRST_TOKEN` | Take the embedding of only the first `[CLS]` token |
| `SEP_TOKEN`, `LAST_TOKEN` | Take the embedding of only the last `[SEP]` token |
| `None` | No reduction is applied and a matrix of embeddings per word in the sentence is returned |

## Roadmap
This extension is still experimental. Possible future updates include:
* Getting document representation from other state-of-the-art NLP models other than Google's BERT.
* Method for computing similarity between `Doc`, `Span` and `Token` objects using the `bert_repr` tensor.
* Getting representation from multiple / other layers in the models.