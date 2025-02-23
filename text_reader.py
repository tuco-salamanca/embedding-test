import pandas as pd

class TextReader:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        #df = pd.read_csv("chat_logs/demo.csv")

    def get_n_rows(self, n: int):
        chat_log = []
        for col2, col3 in zip(self.df.iloc[-n:,1], self.df.iloc[-n:,2]):
            chat_log.append(col3)
        return chat_log