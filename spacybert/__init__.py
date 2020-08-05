from .about import __version__
import logging
from spacy.tokens import Doc, Span, Token
import torch
from transformers import BertModel, BertTokenizer
from typing import Dict, NewType


LANG_ISO_639_1 = NewType('LANG_ISO_639_1', str)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__.split('/')[-1])


class BertInference:
    name = 'bert_inference'

    def __init__(
            self, *, from_pretrained: str = 'bert-base-uncased',
            attr_name: str = 'bert_repr', max_seq_len: int = 512,
            pooling_strategy: str = 'REDUCE_MEAN',
            set_extension: bool = True, force_extension: bool = True):
        """
        Get Bert sentence embeddings using spaCy.

        Keyword arguments only!

        Params
        ------
        from_pretrained: str
            Path to Bert model directory or name of HuggingFace transformers
            pre-trained Bert weights, e.g., 'bert-base-uncased'

        attr_name: str (default='bert_repr')
            Name of the BERT embedding attribute to set to the ._ property.

        max_seq_len: int (default=512)
            Max sequence length for input to Bert.

        pooling_strategy: str (default='REDUCE_MEAN')
            Strategy to generate single sentence embedding from multiple
            word embeddings. Can be one of the following options:

            'REDUCE_MEAN':
                Element-wise average the word embeddings.
            'REDUCE_MAX':
                Element-wise maximum of the word embeddings.
            'REDUCE_MEAN_MAX':
                Apply both 'REDUCE_MEAN' and 'REDUCE_MAX' and concatenate.
                So if the original word embedding is of dimensions (768,),
                then the output will have shape (1536,).
            'CLS_TOKEN' or 'FIRST_TOKEN':
                Take the embedding of only the first [CLS] token.
            'SEP_TOKEN' or 'LAST_TOKEN':
                Take the embedding of only the last [SEP] token.
            None:
                No reduction is applied and a matrix of embeddings per word
                in the sentence is returned.

        set_extension: bool (default=True)
            If True, then 'bert_repr' is set as a property extension for the
            Doc, Span and Token spacy objects. If False, the 'bert_repr' is
            set as an attribute extension with a default value (=None)
            which gets filled correctly when called in a pipeline.

            Set it to False if you want to use this extension in a spacy pipeline.

        force_extension: bool (default=True)
            A boolean value to create the same 'Extension Attribute' upon being
            executed again
        """
        assert max_seq_len > 0
        pooling_strategy = pooling_strategy.upper() if isinstance(pooling_strategy, str) else None
        assert pooling_strategy in (
            None, "REDUCE_MEAN", "REDUCE_MAX", "REDUCE_MEAN_MAX"
            , "CLS_TOKEN", "FIRST_TOKEN", "SEP_TOKEN", "LAST_TOKEN")

        # initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(from_pretrained)

        # initialize the BERT model object
        self.bert_layer = BertModel.from_pretrained(from_pretrained).to(device)

        self.attr_name = attr_name
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.set_extension = set_extension

        if set_extension:
            Doc.set_extension(attr_name, getter=self.__call__, force=force_extension)
            Span.set_extension(attr_name, getter=self.__call__, force=force_extension)
            Token.set_extension(attr_name, getter=self.__call__, force=force_extension)
        else:
            Doc.set_extension(attr_name, default=None, force=force_extension)
            Span.set_extension(attr_name, default=None, force=force_extension)
            Token.set_extension(attr_name, default=None, force=force_extension)

    def __call__(self, doc):
        # preprocessing the text to be suitable for BERT
        # tokenize the sentence
        tokens = self.tokenizer.tokenize(str(doc))

        # inserting the CLS and SEP token in the beginning and end of the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        sep_index = len(tokens) - 1
        if len(tokens) < self.max_seq_len:
            # padding sentences
            tokens = tokens + ['[PAD]' for _ in range(self.max_seq_len - len(tokens))]
        else:
            # pruning the list to be of specified max length
            tokens = tokens[:self.max_seq_len - 1] + ['[SEP]']

        # obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # converting the list to a pytorch tensor
        tokens_ids_tensor = torch.tensor(tokens_ids).unsqueeze(0).to(device)

        # obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long().unsqueeze(0).to(device)

        # feeding the input to BERT model to obtain contextualized representations
        doc_repr = self.bert_layer(
            tokens_ids_tensor, attention_mask=attn_mask)[0].squeeze(0)

        # pool the representation
        if self.pooling_strategy == "REDUCE_MEAN":
            doc_repr = torch.mean(doc_repr, dim=0)
        elif self.pooling_strategy == "REDUCE_MAX":
            doc_repr = torch.max(doc_repr, dim=0)[0]
        elif self.pooling_strategy == "REDUCE_MEAN_MAX":
            doc_repr = torch.cat((
                torch.mean(doc_repr, dim=0)
                , torch.max(doc_repr, dim=0)[0]), dim=-1)
        elif self.pooling_strategy in ("CLS_TOKEN", "FIRST_TOKEN"):
            doc_repr = doc_repr[0]
        elif self.pooling_strategy in ("SEP_TOKEN", "LAST_TOKEN"):
            doc_repr = doc_repr[sep_index]

        if not self.set_extension:
            doc._.set(self.attr_name, doc_repr)
            return doc

        return doc_repr


class MultiLangBertInference:

    name = 'multi_lang_bert_inference'
    models : Dict[LANG_ISO_639_1, BertInference] = dict()

    def __init__(
            self, *, from_pretrained: Dict[LANG_ISO_639_1, str],
            attr_name: str = 'bert_repr', max_seq_len: int = 512,
            pooling_strategy: str = 'REDUCE_MEAN',
            set_extension: bool = True, force_extension: bool = True):
        """
        Use with spacy pipeline for getting BERT (PyTorch) language-specific
        tensor representations when multiple languages are present in the
        dataset and the languages are pre-known.

        Use after applying spacy_langdetect.LanguageDetector() (in pipeline)
        # https://spacy.io/universe/project/spacy-langdetect

        Keyword arguments only!

        Params
        ------
        from_pretrained: Dict[LANG_ISO_639_1, str]
            Mapping between two-letter language codes to path to model
            directory or HuggingFace transformers pre-trained Bert weights.

        attr_name: str (default='bert_repr')
            Same as in BertInference.

        max_seq_len: int (default=512)
            Same as in BertInference.

        pooling_strategy: str (default='REDUCE_MEAN')
            Same as in BertInference.

        set_extension: bool (default=True)
            Same as in BertInference.

        force_extension: bool (default=True)
            Same as in BertInference.
        """
        self.from_pretrained = from_pretrained
        self.attr_name = attr_name
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.set_extension = set_extension
        self.force_extension = force_extension

        if set_extension:
            Doc.set_extension(attr_name, getter=self.__call__, force=force_extension)
            Span.set_extension(attr_name, getter=self.__call__, force=force_extension)
            Token.set_extension(attr_name, getter=self.__call__, force=force_extension)
        else:
            Doc.set_extension(attr_name, default=None, force=force_extension)
            Span.set_extension(attr_name, default=None, force=force_extension)
            Token.set_extension(attr_name, default=None, force=force_extension)

    def _get_model(self, language: str):
        if language not in self.models:
            logger.info(f'load BERT for {language}')
            model_path = self.from_pretrained.get(language, None)
            if not model_path:
                Doc.set_extension(f'{self.attr_name}_{language}', default=None, force=self.force_extension)
                Span.set_extension(f'{self.attr_name}_{language}', default=None, force=self.force_extension)
                Token.set_extension(f'{self.attr_name}_{language}', default=None, force=self.force_extension)
                raise ValueError(f'BERT for language {language} not available.')
            self.models[language] = BertInference(
                from_pretrained=model_path, attr_name=f'{self.attr_name}_{language}',
                max_seq_len=self.max_seq_len,
                pooling_strategy=self.pooling_strategy, set_extension=False)
        return self.models[language]

    def __call__(self, doc):
        # Get the document language
        try:
            lang: LANG_ISO_639_1 = doc._.language['language']
        except:
            raise ValueError('_.language property missing for input doc. Use after spacy_langdetect.LanguageDetector() in nlp.pipe')

        # Load the language-specific model
        # and get the Bert embedding
        try:
            model = self._get_model(lang)
            doc = model(doc)
        except:
            # if language is not available or some other issue,
            # set default value
            doc._.set(f'{self.attr_name}_{lang}', None)
        doc_repr = getattr(doc._, f'{self.attr_name}_{lang}')

        if not self.set_extension:
            doc._.set(self.attr_name, doc_repr)
            return doc

        return doc_repr

# 
# if __name__ == '__main__':
#     import spacy
#     from spacy_langdetect import LanguageDetector

#     texts = [
#         "This is a test",
#         "Dit is een test"
#     ]
#     nlp = spacy.load('nl')
#     nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
#     bert = MultiLangBertInference(
#         from_pretrained={
#             'en': '/path/to/DeepPavlov/bert-base-cased-conversational',
#             'nl': '/path/to/wietsedv/bert-base-dutch-cased'
#         },
#         set_extension=False)
#     nlp.add_pipe(bert, after='language_detector')
#     for doc in nlp.pipe(texts):
#         print(doc._.language)
#         print(doc._.bert_repr)
#     import IPython ; IPython.embed() ; exit()
# 