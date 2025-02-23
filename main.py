from custom_embedding import CustomEmbeddings
from text_reader import TextReader
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

text_reader = TextReader(file_path="chat_logs/demo.csv")

chat_log = text_reader.get_n_rows(10000)

embedding_model = CustomEmbeddings()

print("🔄 임베딩 생성 중...")
embedded_texts = []
for text in tqdm(chat_log, desc="임베딩 진행률", unit="문장"):
    embedded_texts.append(embedding_model.embed_query(text))

print("✅ 모든 임베딩 생성 완료!")

chat_vectors = np.array(embedded_texts)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(chat_vectors)

cluster_topics = []
for i in range(k):
    cluster_texts = [chat_log[idx] for idx in np.where(labels == i)[0]]
    most_common_words = Counter(" ".join(cluster_texts).split()).most_common(5)
    cluster_topics.append(" ".join([word for word, _ in most_common_words]))

# 결과 출력
print(f"🔍 자주 사용된 단어: {cluster_topics}")

#retriever = vectorstore.as_retriever()
#retrieved_documents = retriever.invoke("이 채팅에서 주로 어떤 주제로 이야기를 나누고있어?")
#print("🔍 검색 결과:", retrieved_documents[0].page_content)

