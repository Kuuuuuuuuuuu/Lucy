import speech_recognition as sr


r = sr.Recognizer()
with sr.Microphone() as source: # source = 음성
    print("말해")
    audio = r.listen(source) # 마이크로부터 음성 수신해서 audio 에 저장

try: # 구글 API 하루 50회 제한 문제점 : 긴문장은 끈임없ㅇ이 말해야함. 어.. 음.. 등의 끌림을 인식어려워함
    text = r.recognize_google(audio, language= 'ko') # 수신 음성, 음성 언어 선택
    print(text)
except sr.UnknownValueError: # 음성인식 실패시 처리 구문
    print("뭐?")
except sr.RequestError:
    print('요청실패 : {0}'.format(0)) # 실패 원인에 대한 에러 구문 출력, 키 오류, 네트워크 단절 등
# text 가 번역된 음성