import os
import torch
import json
from transformers import BertConfig, CONFIG_NAME, BertTokenizer, BertModel
from document_bert_architectures import Bert_DOC_TOK, BERT_SEG
from evaluate import evaluation
from encoder import encode_documents
from data import asap_essay_lengths, fix_score


class DocumentBertScoringModel():
    def __init__(self, args=None):
        if args is not None:
            self.args = vars(args)
        self.prompt = int(args.prompt[1])
        chunk_sizes_str = self.args['chunk_sizes']
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)
        bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        # print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        # print("chunk_sizes_str:%s, bert_batch_size_str:%s" % (chunk_sizes_str, bert_batch_size_str))
        if args.bert_init == True:
            self.config, self.bert_tokenizer, self.bert_reg_doc_tok, self.bert_reg_seg = self.init_model()
        else:
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['model_directory'])
            if os.path.exists(self.args['model_directory']):
                if os.path.exists(os.path.join(self.args['model_directory'], CONFIG_NAME)):
                    config = BertConfig.from_json_file(os.path.join(self.args['model_directory'], CONFIG_NAME))
            else:
                config = BertConfig.from_pretrained(self.args['model_directory'])
            self.config = config
            self.bert_reg_doc_tok = Bert_DOC_TOK.from_pretrained(
                self.args['model_directory'] + "/word_document",
                config=config
            )
            self.bert_reg_seg = BERT_SEG.from_pretrained(
                self.args['model_directory'] + "/chunk",
                config=config)
    def _eval(self):
        self.bert_reg_doc_tok.eval()
        self.bert_reg_seg.eval()

    def _train(self):
        self.bert_reg_doc_tok.train()
        self.bert_reg_seg.train()              
    
    def tokenizing(self, data):
        if len(data) == 2:
            document_representations_word_document, document_sequence_lengths_word_document = encode_documents(
                data[0], self.bert_tokenizer, max_input_length=512)
            document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
            for i in range(len(self.chunk_sizes)):
                document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                    data[0],
                    self.bert_tokenizer,
                    max_input_length=self.chunk_sizes[i])
                document_representations_chunk_list.append(document_representations_chunk)
                document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)
        return document_representations_word_document, document_representations_chunk_list

    def eval_all(self, data):
        documents_doc_tok, documents_seg = self.tokenizing(data)
        correct_output = torch.tensor(data[1], dtype=torch.float)
        device = self.args['device']
        self.bert_reg_doc_tok.to(device=device)
        self.bert_reg_seg.to(device=device)
        self._eval()

        with torch.no_grad():
            predictions = torch.empty((documents_doc_tok.shape[0]))
            for i in range(0, documents_doc_tok.shape[0], self.args['batch_size']):
                b_documents_doc_tok = documents_doc_tok[i:i + self.args['batch_size']].to(device=device)
                pred_doc_tok = self.bert_reg_doc_tok(b_documents_doc_tok, device=device)
                pred_doc_tok = torch.squeeze(pred_doc_tok)

                pred_doc_toc_seg = pred_doc_tok
                for chunk_index in range(len(self.chunk_sizes)):
                    batch_document_seg = documents_seg[chunk_index][i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    pred_seg = self.bert_reg_seg(
                        batch_document_seg,
                        device=self.args['device'],
                        bert_batch_size=self.bert_batch_sizes[chunk_index]
                    )
                    pred_seg = torch.squeeze(pred_seg)
                    pred_doc_toc_seg = torch.add(pred_doc_toc_seg, pred_seg)
                predictions[i:i + self.args['batch_size']] = pred_doc_toc_seg
        assert correct_output.shape == predictions.shape

        predictions = predictions.cpu().numpy()
        correct_output = correct_output.cpu().numpy()
        prediction_scores = [fix_score(item, self.prompt) for item in predictions]
        test_eva_res = evaluation(correct_output, prediction_scores)
        # print("pearson:", float(test_eva_res[7]))
        # print("qwk:", float(test_eva_res[8]))
        return float(test_eva_res[7]), float(test_eva_res[8])
    
    def predict(self,batched_data):
        documents_doc_tok, documents_seg = self.tokenizing(batched_data)
        
        device = self.args['device']
        self.bert_reg_doc_tok.to(device=device)
        self.bert_reg_seg.to(device=device)
        self._train()

        predictions = torch.empty((documents_doc_tok.shape[0]))
        documents_doc_tok = documents_doc_tok.to(device=self.args['device'])
        pred_doc_tok = self.bert_reg_doc_tok(documents_doc_tok, device=device)
        pred_doc_tok = torch.squeeze(pred_doc_tok)

        pred_doc_toc_seg = pred_doc_tok
        for chunk_index in range(len(self.chunk_sizes)):
            document_seg_sub = documents_seg[chunk_index]
            document_seg_sub = document_seg_sub.to(device=self.args['device'])
            pred_seg = self.bert_reg_seg(
                document_seg_sub,
                device=self.args['device'],
                bert_batch_size=self.bert_batch_sizes[chunk_index]
            )
            pred_seg = torch.squeeze(pred_seg)
            pred_doc_toc_seg = torch.add(pred_doc_toc_seg, pred_seg)

        return pred_doc_toc_seg

    def init_model(self):
        BASE_MODEL_NAME = 'bert-base-uncased'
        default_config = {
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "type_vocab_size": 2,
            "vocab_size": 30522
            }
        config = BertConfig.from_dict(default_config)
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
        bert_regression_by_word_document = Bert_DOC_TOK.from_pretrained(BASE_MODEL_NAME,config=config)
        bert_regression_by_chunk = BERT_SEG.from_pretrained(BASE_MODEL_NAME,config=config)

        return config, tokenizer, bert_regression_by_word_document,bert_regression_by_chunk
    
    def freeze_bert(self):
        for name, param in self.bert_reg_seg.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        for name, param in self.bert_reg_doc_tok.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False

    def parameters(self):
        torch.nn.modules.Module.parameters(self)
        return list(self.bert_reg_doc_tok.parameters()) + list(self.bert_reg_seg.parameters())
    
    def save_model(self):
        #mkdir
        base_model_path = self.args['model_directory'] 
        doc_tok_model_path = self.args['model_directory'] + "/word_document"
        seg_model_path = self.args['model_directory'] + "/chunk"
        if not os.path.exists(base_model_path):
            os.makedirs(base_model_path)
        if not os.path.exists(doc_tok_model_path):
            os.makedirs(doc_tok_model_path)
        if not os.path.exists(seg_model_path):
            os.makedirs(seg_model_path)
        #save tokenizer
        self.bert_tokenizer.save_pretrained(base_model_path)
        #save model as 'pytorch_model.bin'
        torch.save(self.bert_reg_doc_tok.state_dict(), doc_tok_model_path + "/pytorch_model.bin")
        torch.save(self.bert_reg_seg.state_dict(), seg_model_path + "/pytorch_model.bin")
        #save config
        self.config.to_json_file(base_model_path + "/config.json")

if __name__ == "__main__":
    pass

    # p = configargparse.ArgParser(default_config_files=["asap.ini"])
    # args = _initialize_arguments(p)
    # args.is_train = True
    # model = DocumentBertScoringModel(args=args)


   