import socket

# 서버 주소 및 포트 설정
HOST = 'IP 주소' # 서버가 바인딩할 IP 주소 (로컬 호스트)
PORT = '포트 번호' # 서버가 바인딩할 포트 번호

# 소캣 생성 (IPv4, TCP 방식)
nano_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 서버에 연결 요청 (블로킹 상태, 서버가 응답할 때 까지 대기)
nano_socket.connect((HOST, PORT))

while True:
    # 사용자가 입력을 받아 메시지를 전송 (블로킹 상태)
    message = input("메시지를 입력 ( 종료 : exit ):")

    # 사용자가 exit를 입력을 받으면 종료
    if message.lower == 'exit' :
        break

    # 입력받능 문자열으 UTF-8로 인코딩하여 서버에 전송
    # 블로킹 상태 (모든 데이터 전송 후 다음 코드 실행 )
    nano_socket.sendall(message.encode())

    # 서버로부터 응답 데이터 수신 ( 최대 1024 바이트, 블로킹 상태 )
    car_entry_data = nano_socket.recv(1024)

    # 수신한 데이터를 디코딩하여 출력
    print(f"서버 응답 : {car_entry_data.decode()}")

# 클라이언트 소켓 종료 ( 서버와의 연결 해제 )
nano_socket.close()