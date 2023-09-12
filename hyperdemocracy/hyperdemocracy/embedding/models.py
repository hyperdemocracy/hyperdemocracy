from FlagEmbedding import FlagModel

class BGESmallEn:
    model_name = "BAAI/bge-small-en"
    
    def __init__(self):
        self.model = FlagModel(self.model_name, query_instruction_for_retrieval="Represent this sentence for searching relevant passages:")

    