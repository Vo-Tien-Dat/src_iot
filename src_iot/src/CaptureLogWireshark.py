import subprocess

command = 'dumpcap -i "Local Area Connection* 10" -c 1000 -b "filesize:1000000" -b "files:2" -w E:/log/captures/captures_cic_2023.pcap'

try:
    subprocess.run(command, shell=True, check=True)
    print("Chạy thành công")
except subprocess.CalledProcessError as e:
    print(f"Chạy đang bị lỗi {e.returncode}.")