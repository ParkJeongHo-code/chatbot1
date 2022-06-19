
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import tensorflow_datasets as tfds
import os
import re
import urllib.request
data=pd.read_csv("/Users/bagjeongho/Desktop/chatbot1-main/chatbot_data.csv")
data=data.loc[:20000,:]
#urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
#data = pd.read_csv('ChatBotData.csv')
#SubwordTextEncoder를 사용하기 위해 전처리 과정
q=[]
a=[]
for i in range(data.shape[0]):
    data_q=re.sub(r"([?.!,])", r" \1 ", data.loc[i,'Q'])
    data_q=data_q.strip()
    
    data_a=re.sub(r"([?.!,])", r" \1 ", data.loc[i,'A'])
    data_a=data_a.strip()   
    print(data_a) 
    q.append(data_q)
    a.append(data_a)
print('start tokenizer')
tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(q+a,target_vocab_size=2**13)# 단어 모음집 생성
#문장의 start와 end를 정의할 토큰 정의
start_token,eos_token= [tokenizer.vocab_size],[tokenizer.vocab_size+1]
#문장의 시작과 끝을 가리키는 토큰들을 고려해서 단어장의 사이즈 정의
vocab_size=tokenizer.vocab_size+2
enc_q=[]
enc_a=[]
for i in q:
    enc_q.append(start_token+tokenizer.encode(i)+eos_token)
for j in a:
    enc_a.append(start_token+tokenizer.encode(j)+eos_token)
    
for_max_len=[]
for_max_len.extend(enc_q)
for_max_len.extend(enc_a)

max_len=max(len(i) for i in for_max_len)
sentence_int_pad_q=tf.keras.preprocessing.sequence.pad_sequences(enc_q,maxlen=max_len,padding="post")
sentence_int_pad_a=tf.keras.preprocessing.sequence.pad_sequences(enc_a,maxlen=max_len,padding="post")
print('make dataset')

datasets=tf.data.Dataset.from_tensor_slices(({'inputs':sentence_int_pad_q,'dec_inputs':sentence_int_pad_a[:,:-1]},{"outputs":sentence_int_pad_a[:,1:]}))
print('complete make dataset')
"""
class pos_enc(tf.keras.layers.Layer):
    def __init__(self,vocab_size,d_model):
        super(pos_enc, self).__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        
    def get_pos(self,pos,i,d_model):#pos 행 ,D_MODEL은 임베딩의 차원 
        if i%2==0:
            return tf.math.sin(pos/(10000**(2*i/d_model)))
        else:
            return tf.math.cos(pos/(10000**(2*i/d_model)))
        
    def make_mat(self,vocab_size,d_model,inputs):
        re=[]
        pos_mat=[]
        for i in range(vocab_size):
            row=[]
            for j in range(d_model):
                pos_en= self.get_pos(pos=i+1,i=j+1,d_model=d_model)
                row.append(inputs[i,j].numpy()+pos_en.numpy())
                #row.append(pos_en)
            pos_mat.append(row)
        return pos_mat


    def call(self,inputs):
        pos_mat=self.make_mat(self.vocab_size,self.d_model,inputs)
        return tf.add(inputs,pos_mat)


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)
            
def create_padding_mask(x):#아래 scale_dot_product에서 마스킹을 위한 m을 만드는 함수이다. 아래에서 layers.Lamda를 통해 활용
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
# (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]
    
def scaled_dot_product_attention(q,k,v,m=None):
    att_score=tf.matmul(q,k,transpose_b=True)/tf.math.sqrt(tf.cast(tf.shape(q)[-1],tf.float32))
    if m is not None:
        att_score += (m * -1e9)#거의 -무한대로 마스킹해줌
    att_w=tf.nn.softmax(att_score,axis=-1)
    result=tf.matmul(att_w,v)
    return result

class Multi_head_layer(tf.keras.layers.Layer):#d_model은 입력의 차원 
    def __init__(self,d_model,num_of_head):
        super(Multi_head_layer, self).__init__()
        self.w_q=tf.keras.layers.Dense(d_model)
        self.w_k=tf.keras.layers.Dense(d_model)
        self.w_v=tf.keras.layers.Dense(d_model)
        self.w_summ=tf.keras.layers.Dense(d_model)
        self.depth=int(d_model/num_of_head)
        self.num_head=num_of_head

    def call(self,query,key,value,mask):
        q,k,v=query,key,value
        m=mask
        batch_size=tf.shape(q)[0]
        query=self.w_q(q)
        key=self.w_k(k)
        value=self.w_v(v)
        query=tf.reshape(query,shape=(batch_size,self.num_head,-1,self.depth))
        key=tf.reshape(key,shape=(batch_size,self.num_head,-1,self.depth))
        value=tf.reshape(value,shape=(batch_size,self.num_head,-1,self.depth))
        atten_w=scaled_dot_product_attention(query,key,value,m)
        atten_w = tf.transpose(atten_w, perm=[0, 2, 1, 3])
        atten_w=tf.reshape(atten_w,(batch_size,-1,self.depth))
        atten_w=self.w_summ(atten_w)
        
        return atten_w
#x의 형태는(batch_size,문장의 길이,depth)
def encoder_block(d_model,num_of_head,d_ff,name="encoder_block"):
    input_=tf.keras.layers.Input(shape=(None,d_model))
    #인코더의 padding mask 정의
    padding_mask=tf.keras.layers.Input(shape=(1,1,None),name='padding_mask')
    #아까 정의한 multi head attention을 사용한다.
    attention=Multi_head_layer(d_model,num_of_head)(input_,input_,input_,padding_mask)
    #multi head attention의 결과를 잔차 연결한다
    attention=tf.add(input_,attention)
    #layernormalization을 해준다.
    attention=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention)
    #여기가 feed forward net부분이다.
    attention_1=tf.keras.layers.Dense(d_ff,activation='relu')(attention)
    attention_1=tf.keras.layers.Dense(d_model)(attention)
    #그후 잔차 연결과 layernormalization을 해준다.
    attention_2=tf.add(attention_1,attention)
    outputs=tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_2)
    
    return tf.keras.Model(inputs=[input_,padding_mask],outputs=outputs)


def decoder_block(d_model,d_ff,num_of_head,name="decoder_block"):#d_model은 출력의 차원
    input_=tf.keras.layers.Input(shape=(None,d_model))
    enc_output=tf.keras.layers.Input(shape=(None,d_model))
    
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")#이거는 현재 시점보다 뒤에 시점의 attention을 배제하기 위해 masking하는 부분
    padding_mask=tf.keras.layers.Input(shape=(1,1,None))#<pad>토큰을 masking해줘서 유사도를 구하지 않게 해줌
    
    #self attention부분이다 self attention이기에 현재 이후의 시점의 attention을 배제하기 위해 look_ahead_mask를 사용한다
    self_attention=Multi_head_layer(d_model,num_of_head)(input_,input_,input_,look_ahead_mask)
    self_attention=tf.add(self_attention,input_)
    self_attention=tf.keras.layers.LayerNormalization()(self_attention)
    
    enc_dec_attention=Multi_head_layer(d_model,num_of_head)(self_attention,enc_output,enc_output,padding_mask)
    enc_dec_attention=tf.add(self_attention,enc_dec_attention)
    enc_dec_attention=tf.keras.layers.LayerNormalization()(enc_dec_attention)
    
    dec_attention_1=tf.keras.layers.Dense(d_ff,activation='relu')(enc_dec_attention)
    dec_attention_1=tf.keras.layers.Dense(d_model)(enc_dec_attention)
    
    last_attention=tf.add(dec_attention_1,enc_dec_attention)
    last_attention=tf.keras.layers.LayerNormalization()(enc_dec_attention)
    
    return tf.keras.models.Model(inputs=[input_,enc_output,padding_mask,look_ahead_mask],outputs=last_attention)

def transformer(d_model,d_ff,num_of_head,vocab_size):
    enc_input=tf.keras.layers.Input(shape=(None,))
    dec_input=tf.keras.layers.Input(shape=(None,))
    
    emb_enc=tf.keras.layers.Embedding(vocab_size,d_model)(enc_input)
    emb_dec=tf.keras.layers.Embedding(vocab_size,d_model)(dec_input)
    
    pad_mask=tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='enc_padding_mask')(enc_input)
    
    look_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None),name='look_ahead_mask')(dec_input)


    pos_enc_enc=pos_enc(vocab_size,d_model)(emb_enc)

    pos_enc_dec=pos_enc(vocab_size,d_model)(emb_dec)

    
    first_enc_out=encoder_block(d_model,num_of_head,d_ff)([pos_enc_enc,pad_mask])
    second_enc_out=encoder_block(d_model,num_of_head,d_ff)([first_enc_out,pad_mask])
    third_enc_out=encoder_block(d_model,num_of_head,d_ff)([second_enc_out,pad_mask])
    four_enc_out=encoder_block(d_model,num_of_head,d_ff)([third_enc_out,pad_mask])
    
    first_dec_out=decoder_block(d_model,d_ff,num_of_head)([pos_enc_dec, four_enc_out, pad_mask, look_mask])
    second_dec_out=decoder_block(d_model,d_ff,num_of_head)([first_dec_out, four_enc_out, pad_mask, look_mask])
    third_dec_out=decoder_block(d_model,d_ff,num_of_head)([second_dec_out, four_enc_out, pad_mask, look_mask])
    four_dec_out=decoder_block(d_model,d_ff,num_of_head)([third_dec_out, four_enc_out, pad_mask, look_mask])
    
    outputs = tf.keras.layers.Dense(vocab_size, name="outputs")(four_dec_out)
    return tf.keras.models.Model(inputs=[enc_input,dec_input],outputs=outputs)
"""
class PositionalEncoding(tf.keras.Model):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
        sines = tf.math.sin(angle_rads[:, 0::2])

        # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장 길이)

    # Q와 K의 곱. 어텐션 스코어 행렬.
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # 스케일링
    # dk의 루트값으로 나눠준다.
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
    # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
    if mask is not None:
        logits += (mask * -1e9)

    # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
    # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights
class MultiHeadAttention(tf.keras.Model):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model을 num_heads로 나눈 값.
        # 논문 기준 : 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = tf.keras.layers.Dense(units=d_model)

    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. 헤드 연결(concatenate)하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))

        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)

        return outputs
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]
def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention")({
            'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
            'mask': padding_mask # 패딩 마스크 사용
        })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(inputs + attention)

    # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 인코더는 패딩 마스크 사용
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 인코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, padding_mask], outputs=outputs, name=name)
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
    return tf.maximum(look_ahead_mask, padding_mask)
def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention_1")(inputs={
            'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
            'mask': look_ahead_mask # 룩어헤드 마스크
        })

    # 잔차 연결과 층 정규화
    attention1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention1 + inputs)

    # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention_2")(inputs={
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
            'mask': padding_mask # 패딩 마스크
        })

    # 드롭아웃 + 잔차 연결과 층 정규화
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(attention2 + attention1)

    # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # 드롭아웃 + 잔차 연결과 층 정규화
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

    # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
    look_ahead_mask = tf.keras.Input(
        shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

    # 포지셔널 인코딩 + 드롭아웃
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # 디코더를 num_layers개 쌓기
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
            dropout=dropout, name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return tf.keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)
def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 인코더의 패딩 마스크
    enc_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='enc_padding_mask')(inputs)

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # 디코더의 패딩 마스크(두번째 서브층)
    dec_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None),
        name='dec_padding_mask')(inputs)

    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
        d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
        d_model=d_model, num_heads=num_heads, dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # 다음 단어 예측을 위한 출력층
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
BATCH_SIZE = 32
BUFFER_SIZE = 20000
print('dataset additional processing')
datasets = datasets.cache()
datasets = datasets.shuffle(BUFFER_SIZE)
datasets = datasets.batch(BATCH_SIZE)
datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)

print('dataset additional processing complete')
print(vocab_size)
NUM_LAYERS = 1
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

small_transformer = transformer(
    vocab_size=vocab_size,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

print(small_transformer.summary())
optimizer=tf.keras.optimizers.Adam(beta_1=0.9,beta_2=0.98,epsilon=1e-9)

def accuracy(y_true,y_pred):
    #레이블의 크기는 (batch_size, MAX_LENGTH-1)
    y_true=tf.reshape(y_true,shape=(-1,max_len-1))
    return tf.keras.metrics.sparse_categorical_accuracy(y_true,y_pred)
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)
learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
small_transformer.compile(optimizer=optimizer,loss=loss_function,metrics=[accuracy])
EPOCHS = 100
for epo in range(EPOCHS):
    small_transformer.fit(datasets, epochs=1, verbose=1)
    small_transformer.save_weights('/Users/bagjeongho/Desktop/chatbot1-main/cahtbot_w.h5',save_format="h5")

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(
        start_token + tokenizer.encode(sentence) + eos_token, axis=0)

    output = tf.expand_dims(start_token, 0)

    # 디코더의 예측 시작
    for i in range(max_len):
        predictions = small_transformer(inputs=[sentence, output], training=False)

        # 현재(마지막) 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, eos_token[0]):
            break

        # 마지막 시점의 예측 단어를 출력에 연결한다.
        # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence
output = predict("딥 러닝 자연어 처리를 잘 하고 싶어")
