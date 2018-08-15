# 파일은 두가지 모드로 작동합니다.
# 터미널에서 python3 csv2utf8.py [파일이름] 으로 명령어를 입력하면 바로 utf-8로 인코딩된 파일을 생성합니다.
# 인자 없이 명령어를 입력하면 파일이름을 입력해 달라는 input() 프롬프트가 보입니다.

import os
import csv
import sys

# EUC.kr로 인코딩된 파일을 인자로 받아서 열고 다시  changedEuckr.csv 파일로 쓴다.
#if len(sys.argv) > 1:
#    euckrFile = sys.argv[1]
#else:
euckrFile = input("file 이름 :")
fileName = euckrFile.split('.')[0]
resultName = fileName + '-utf8.csv'
#resultName = fileName

# 결과파일(changedEuckr.csv 파일이 있다면 일단 삭제한다
if os.path.isfile(resultName):
    os.remove(resultName)

path = os.getcwd()
resultPath = os.path.join(path, resultName)
openPath = os.path.join(path, euckrFile)
with open(openPath, mode='r', encoding='euc-kr') as euckr:
    reading = csv.reader(euckr)
    for line in reading:
        text = ''
        count = len(line)
        for idx, one in enumerate(line):
            if ( idx is not count-1):
                text = text + one + ','
            else:
                text += one
        text += '\n'
        utf8 = open(resultPath, mode='a', encoding='utf-8')
        utf8.write(text)
        utf8.close()

# 파일 쓰기가 완료되었다는 안내를 보여준다.
print("인코딩 변경된 파일이 새롭게 작성되었습니다.")
print(resultName + " 파일을 확인하세요")
