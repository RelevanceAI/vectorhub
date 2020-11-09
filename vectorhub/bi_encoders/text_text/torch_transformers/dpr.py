from ..base import BaseTextText2Vec
from ....doc_utils import ModelDefinition
from ....import_utils import *
from datetime import date
if is_all_dependency_installed('encoders-text-torch-transformers'):
    from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRReader, DPRReaderTokenizer
    import torch
    import numpy as np

DPRModelDefinition = ModelDefinition(markdown_filepath='bi_encoders/text_text/torch_transformers/dpr')
__doc__ = DPRModelDefinition.create_docs()

class DPR2Vec(BaseTextText2Vec):
    definition = DPRModelDefinition
    def __init__(self):
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        self.context_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', return_dict=True)

        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.query_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        self.reader_tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
        self.reader_model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base', return_dict=True)
        self.vector_length = 768

    def encode_question(self, question):
        """
            Encode a question with DPR.
        """
        input_ids = self.query_tokenizer(question, return_tensors='pt')["input_ids"]
        return self.query_encoder(input_ids)[0].tolist()[0]

    def bulk_encode_questions(self, questions: str):
        """
            Bulk encode the question
        """
        input_ids = self.query_tokenizer(questions, truncation=True, max_length=True, return_tensors='pt')["input_ids"]
        return self.query_encoder(input_ids)[0].tolist()

    def encode_answer(self, answer: str):
        """
            Encode an answer with DPR.
        """
        if isinstance(answer, str):
            input_ids = self.context_tokenizer(answer, return_tensors='pt', truncation=True, 
            max_length=512)["input_ids"]
            return self.context_model(input_ids).pooler_output.tolist()[0]
        elif isinstance(answer, list):
            return self.bulk_encode_answers(answer)

    def bulk_encode_answers(self, answers: str):
        """
            Bulk encode the answers with DPR.
        """
        input_ids = self.context_tokenizer(answers, return_tensors='pt', truncation=True, padding=True,
        max_length=512)["input_ids"]
        return self.context_model(input_ids).pooler_output.tolist()
