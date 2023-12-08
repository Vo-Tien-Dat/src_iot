import psutil

def get_interface_names():
    interfaces = psutil.net_if_addrs()
    interface_names = []
    for interface_name, interface_info in interfaces.items():
        interface_names.append(interface_name)
    return interface_names



if __name__ == "__main__":
    interface_names = get_interface_names()
    print(interface_names)