import socket

# 서버 설정
HOST = 'IP 주소' # 서버가 바인딩할 IP 주소 (로컬 호스트)
PORT = '포트 번호' # 서버가 바인딩할 포트 번호

# 소캣 생성 (IPv4, TCP 방식)
orin_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 소캣 바인딩
orin_socket.bind((HOST, PORT))

# 연결 대기
# 클라이언트의 접속을 기다림( 최대 5개 대기 가능 )
orin_socket.listen(5)
print(f"서버가 {HOST}:{PORT}에서 대기 중...")

while True:
    # 클라이언트로부터 데이터 수신
    # 새로운 소켓과 클라이언트 주소 반환 ( 블로킹 상태 )
    nano_socket, nano_address = orin_socket.accept()
    print(f"클라이언트 {nano_address} 연결됨")

    while True:
        # 클라이언트로 부터 데이터 수신
        # 최대 1024바이트, 블로킹 상태
        car_entry_data = nano_socket.recv(10)

        # 클라이언트가 연결을 종료하면 루트 탈출
        if not car_entry_data:
            break

        # 받은 데이터를 문자열로 변환하여 출력
        print(f"받은 데이터: {car_entry_data.decode()}")

        # 받은 데이터를 클라이언트에게 그대로 다시 전송 ( 에코 서버 )
        # 블로킹 상태 ( 모든 데이터 전송 후 다음 코드 실행 )
        nano_socket.sendall(car_entry_data)

    # 클라이언트 소캣 종료
    nano_socket.close()
