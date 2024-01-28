from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

pretrained_name = "mdhugol/indonesia-bert-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(pretrained_name)
tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}



def sentimen(text):
    result = sentiment_analysis(text)
    status = label_index[result[0]['label']]
    score = result[0]['score']
    return f'Label: {status} ({score * 100:.3f}%)'


if __name__ == '__main__':
    pos_text = "Sangat bahagia hari ini"
    neg_text = "Dasar anak sialan!! Kurang ajar!!"
    pesan = "saya Jenderal saya ikut berkali-kali dalam aksi-aksi pertempuran saya saksi Saya melihat pemimpin yang bisa ambil keputusan"
    result= sentimen(pesan)
    print(result)


