from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.llms.llm import LLM
from llama_index.core.settings import Settings
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from typing import Optional, Type, List, Any
from llama_index.core.llms.utils import LLMType, resolve_llm
import numpy as np

import spacy
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
import re
from nltk.corpus import wordnet
import random
import nltk

class CustomPosRewritingQueryEngine(CustomQueryEngine):
    """Custom Pos Rewriting Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: Optional[BaseSynthesizer] = None
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None
    callback_manager: Optional[CallbackManager] = None
    text_qa_template: Optional[BasePromptTemplate] = None
    pos_rewriting_threshold: Optional[float] = None
    number_words_substitute: float = 0.0
    number_questions: int = 0
    translate_query: bool = False
    translate_context: bool = False

    @classmethod
    def from_args(
        cls,
        retriever: BaseRetriever,
        llm: Optional[LLM] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        output_cls: Optional[Type[BaseModel]] = None,
        use_async: bool = False,
        streaming: bool = False,
        # add int optional parameter named pos_rewriting_threshold
        pos_rewriting_threshold: Optional[float] = None,
        translate_query: bool = False,
        translate_context: bool = False,
        **kwargs: Any,
    ) -> "CustomPosRewritingQueryEngine":
        """Initialize a CustomPosRewritingQueryEngine object."""
        llm = llm or Settings.llm

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            llm=llm,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            summary_template=summary_template,
            simple_template=simple_template,
            response_mode=response_mode,
            output_cls=output_cls,
            use_async=use_async,
            streaming=streaming,
        )

        callback_manager = callback_manager or Settings.callback_manager

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=callback_manager,
            node_postprocessors=node_postprocessors,
            text_qa_template=text_qa_template,
            pos_rewriting_threshold=pos_rewriting_threshold,
            translate_query=translate_query,
            translate_context=translate_context,
        )

    # Função para encontrar sinônimo
    def get_synonym(self, word):
        synsets = wordnet.synsets(word)
        if synsets:
            synonyms = [lemma.name() for syn in synsets for lemma in syn.lemmas()]
            synonyms = list(set(synonyms))  # Remover duplicatas
            if word in synonyms:  # Remover a palavra original
                synonyms.remove(word)
            return random.choice(synonyms) if synonyms else word
        return word

    # Função para substituir palavras principais
    def replace_main_words(self, sentence, percentage):

        # Baixe os dados necessários apenas uma vez
        nltk.download('wordnet', quiet=True)
        # Carregue o modelo do spaCy
        nlp = spacy.load("en_core_web_sm")

        infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r'(?<=[0-9])[+\-\*^](?=[0-9-])',
                r'(?<=[{al}{q}])\.(?=[{au}{q}])'.format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                # REMOVE: commented out regex that splits on hyphens between letters:
                #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                # EDIT: remove split on slash between letters, and add comma
                #r'(?<=[{a}0-9])[:<>=/](?=[{a}])'.format(a=ALPHA),
                r'(?<=[{a}0-9])[:<>=,](?=[{a}])'.format(a=ALPHA),
                # ADD: ampersand as an infix character except for dual upper FOO&FOO variant
                r'(?<=[{a}0-9])[&](?=[{al}0-9])'.format(a=ALPHA, al=ALPHA_LOWER),
                r'(?<=[{al}0-9])[&](?=[{a}0-9])'.format(a=ALPHA, al=ALPHA_LOWER),
            ]
        )

        custom_suffixes = [r'[-]']
        suffixes = nlp.Defaults.suffixes
        suffixes = tuple(list(suffixes) + custom_suffixes)

        infix_re = spacy.util.compile_infix_regex(infixes)
        suffix_re = spacy.util.compile_suffix_regex(suffixes)

        nlp.tokenizer.suffix_search = suffix_re.search
        nlp.tokenizer.infix_finditer = infix_re.finditer

        # tokenizar a sentença
        doc = nlp(sentence)
        
        # filtrar apenas palavras principais simples (sem traços)
        main_words = [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
        total_main_words = len(main_words)
        #print(f"Main words: {main_words}")
        
        # Determinar quantas palavras substituir
        num_to_replace = max(1, round(percentage * total_main_words))  # Pelo menos 1 palavra
        self.number_words_substitute += num_to_replace
        #print(f"Total main words: {total_main_words}, words to replace: {num_to_replace}")
        
        # Escolher palavras para substituir aleatoriamente
        words_to_replace = random.sample(main_words, min(num_to_replace, total_main_words))
        #print(f"Words to replace: {words_to_replace}")
        
        # Substituir palavras selecionadas por sinônimos
        new_sentence_str = sentence
        new_sentence = []
        for token in doc:
            if token in words_to_replace:
                synonym_str = self.get_synonym(token.text).replace("_", " ")
                new_sentence_str = new_sentence_str.replace(token.text, synonym_str)
                new_sentence.append(synonym_str)
                #print(f"New word '{synonym_str}'")
            else:
                # remove spaces of token.text
                new_sentence.append(token.text.replace(" ", ""))
        return new_sentence_str

    def custom_query(self, query_str: str):
        self.number_questions += 1
        nodes = self.retriever.retrieve(query_str)
        llm_context_list = [n.node.get_content() for n in nodes]

        # translate
        if self.translate_query is True:
            posrewriting_prompt = PromptTemplate(
                "Translate the sentence below to pt-br.\n"
                "Sentence: {query_str}\n"
            )
            response = self.response_synthesizer._llm.complete(
                posrewriting_prompt.format(query_str=query_str)
            )
            query_str = str(response)
            print(f"\nTranslated query: {query_str}")
        
        if self.translate_context is True:
            llm_context_aux = []
            for context_aux in llm_context_list:
                posrewriting_prompt = PromptTemplate(
                    "Translate the text below to pt-br.\n"
                    "Text: {context_str}\n"
                )
                response = self.response_synthesizer._llm.complete(
                    posrewriting_prompt.format(context_str=context_aux)
                )
                llm_context_aux.append(str(response))
            llm_context_list = llm_context_aux
            print(f"\nTranslated context: {llm_context_list[0][:200]}\n\n")

        # substitute main words
        if self.pos_rewriting_threshold is not None:
            # posrewriting_prompt = PromptTemplate(
            #     f"Rewrite the sentence below, changing just {self.pos_rewriting_threshold} word(s) in the sentence to a corresponding synonym.\n"
            #     "Query: {query_str}\n"
            #     "Rewritten query: "
            # )
            # response = self.response_synthesizer._llm.complete(
            #     posrewriting_prompt.format(query_str=query_str)
            # )
            # query_str = str(response)

            query_str_old = query_str
            query_str = self.replace_main_words(query_str, self.pos_rewriting_threshold)
            #print(f"Number questions: {self.number_questions}")
            #print(f"Number word subs: {self.number_words_substitute}")
            print(f"Rate  substitute: {self.number_words_substitute / self.number_questions}")
            #print(f"Original query: {query_str_old}")
            #print(f"Rewritte query: {query_str}\n\n")
        
        # query_bundle = QueryBundle(query_str)
        # response = self.response_synthesizer.synthesize(
        #     query=query_bundle,
        #     nodes=nodes,
        # )
        # return response

        context_str = "\n\n".join(llm_context_list)
        response = self.response_synthesizer._llm.complete(
            self.text_qa_template.format(context_str=context_str, query_str=query_str)
        )
        return {
            "response": str(response),
            "llm_context_list": llm_context_list
        }

    def custom_query_old(self, query_str: str):
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_str}
        ) as query_event:
            nodes = self.retriever.retrieve(query_str)
            #print(f"Nodes: {nodes}")
            query_bundle = QueryBundle(query_str)
            response = self.response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )
            #context_str = "\n\n".join([n.node.get_content() for n in nodes])
            #response = self.llm.complete(
            #    self.text_qa_template.format(context_str=context_str, query_str=query_str)
            #)
            # print(f"RESSSSSS: {response}")
            query_event.on_end(payload={EventPayload.RESPONSE: response})
        # print(f"11111: {response}")
        return response