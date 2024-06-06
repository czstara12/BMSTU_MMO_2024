import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.parse import CoreNLPParser

# 下载必要的资源
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# 示例文本
text = "Apple is looking at buying U.K. startup for $1 billion."

# 1. 分词
tokens = word_tokenize(text)
print("分词结果：", tokens)

# 2. 词性标注
tagged_tokens = pos_tag(tokens)
print("词性标注结果：", tagged_tokens)

# 3. 词形还原
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("词形还原结果：", lemmatized_tokens)

# 4. 命名实体识别
named_entities = ne_chunk(tagged_tokens)
print("命名实体识别结果：")
print(named_entities)

# 5. 句法解析
parser = CoreNLPParser(url='https://corenlp.run/')
parse_tree = next(parser.raw_parse(text))
print("句法解析结果：")
print(parse_tree)
