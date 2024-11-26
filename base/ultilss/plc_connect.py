import socket, time

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas


class PLC_Connection:

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def connect_plc_keyence(self):
        try:
            soc.connect((self.host, self.port))
            return True
        except OSError:
            print("Can't connect to PLC")
            time.sleep(3)
            print("Reconnecting....")
            return False

    def run_plc_keyence(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_keyence(self.host, self.port)
        print("connected")

    def read_plc_keyence(self, data):
        a = "RD "
        c = "\x0D"
        d = a + data + c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        data = soc.recv(1024)
        datadeco = data.decode("UTF-8")
        data1 = int(datadeco)
        return data1

    def socket_connect(self, soc, host, port):
        """
        Thực hiện kết nối với PLC
        host : địa chỉ IP của PLC
        port : port sử dụng bên PLC
        """
        try:
            soc.connect((host, port))
            return True
        except:
            return False

    def readdata(self, soc, data):
        """
        # Thực hiện đọc dữ liệu từ PLC
        data : Thanh ghi bên PLC. Vd : DM1
        """
        a = "RD "
        c = "\x0D"
        d = a + data + c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        response = soc.recv(1024)
        dataFromPLC = response.decode("UTF-8")
        return int(dataFromPLC)

    def writedata(self, soc, register, data):
        """
        Ghi dữ liệu vào PLC
        register : Thanh ghi cần ghi dữ liệu bên PLC
        data : Dữ liệu cần truyền là
        """
        a = "WR "
        b = " "
        c = "\x0D"
        d = a + register + b + str(data) + c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        response = soc.recv(1024)

    def write_plc_keyence(self, register, data):
        a = "WR "
        b = " "
        c = "\x0D"
        d = a + register + b + str(data) + c
        datasend = d.encode("UTF-8")
        soc.sendall(datasend)
        datares = soc.recv(1024)

    def connect_plc_omron(self):
        global fins_instance
        try:
            fins_instance = UDPFinsConnection()
            fins_instance.connect(self.host)
            fins_instance.dest_node_add = 1
            fins_instance.srce_node_add = 25
            return True
        except:
            print("Can't connect to PLC")
            for i in range(100000000):
                pass
            print("Reconnecting....")
            return False

    def run_plc_omron(self):
        connected = False
        while connected == False:
            connected = self.connect_plc_omron(self.host)
            print("connecting ....")
        print("connected plc")

    def read_plc_omron(self, register):
        register = (register).to_bytes(2, byteorder="big") + b"\x00"
        read_var = fins_instance.memory_area_read(
            FinsPLCMemoryAreas().DATA_MEMORY_WORD, register
        )
        read_var = int.from_bytes(read_var[-2:], byteorder="big")
        return read_var

    def write_plc_omron(self, register, data):
        register = (register).to_bytes(2, byteorder="big") + b"\x00"
        fins_instance.memory_area_write(
            FinsPLCMemoryAreas().DATA_MEMORY_WORD, register, b"\x00\x00", data
        )
