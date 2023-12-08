global CONST_DIC_PATH
CONST_DIC_PATH = "../data/"
CONST_FILE_NAMES = [
    'part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00001-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00003-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00004-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00005-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00006-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
    'part-00007-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv',
]

CONST_FILE_PATHS = [ CONST_DIC_PATH + FILE_NAME for FILE_NAME in CONST_FILE_NAMES]

global CONST_FIELDS

global CONST_ORDINARY_LABEL_NAME

global CONST_DDOS_LABEL

global CONST_DOS_LABEL

global CONST_MIRAI_LABEL

global CONST_SPOOFING_LABEL

global CONST_RECON_LABEL

global CONST_WEB_LABEL

global CONST_BRUTE_FORCE_LABEL

global CONST_DROP_FIELD

global CONST_NAME_MODEL

global CONST_TWO_CLASSES_WITH_LSTM

global CONST_TWO_CLASSES_WITH_RFC

global CONST_EIGHT_CLASSES_WITH_LSTM

CONST_NAME_MODEL = 'model.h5'

CONST_FIELDS = [
    'flow_duration',
    'Header_Length',
    'Protocol_Type',
    'Duration',
    'Rate',
    'Srate',
    'Drate',
    'fin_flag_number',
    'syn_flag_number',
    'rst_flag_number',
    'psh_flag_number',
    'ack_flag_number',
    'ece_flag_number',
    'cwr_flag_number',
    'ack_count',
    'syn_count',
    'fin_count',
    'urg_count',
    'rst_count',
    'HTTP',
    'HTTPS',
    'DNS',
    'Telnet',
    'SMTP',
    'SSH',
    'IRC',
    'TCP',
    'UDP',
    'DHCP',
    'ARP',
    'ICMP',
    'IPv',
    'LLC',
    'tot_sum',
    'Min',
    'Max',
    'AVG',
    'Std',
    'Tot_size',
    'IAT',
    'Number',
    'Magnitue',
    'Radius',
    'Covariance',
    'Variance',
    'Weight',
    'label']

## Nhãn bình thường
CONST_ORDINARY_LABEL_NAME = 'BenignTraffic'

## Gồm có 7 nhãn tấn công

CONST_DDOS_LABEL = [
    'DDoS-ICMP_Flood',
    'DDoS-UDP_Flood',
    'DDoS-TCP_Flood',
    'DDoS-PSHACK_Flood',
    'DDoS-SYN_Flood',
    'DDoS-RSTFINFlood',
    'DDoS-SynonymousIP_Flood',
    'DDoS-ICMP_Fragmentation',
    'DDoS-ACK_Fragmentation',
    'DDoS-UDP_Fragmentation',
    'DDoS-HTTP_Flood',
    'DDoS-SlowLoris']

CONST_DOS_LABEL = [
    'DoS-UDP_Flood',
    'DoS-TCP_Flood',
    'DoS-SYN_Flood',
    'DoS-HTTP_Flood']

CONST_MIRAI_LABEL = [
    'Mirai-greeth_flood',
    'Mirai-udpplain',
    'Mirai-greip_flood']

CONST_SPOOFING_LABEL = [
    'MITM-ArpSpoofing',
    'DNS_Spoofing']

CONST_RECON_LABEL = [
    'Recon-HostDiscovery',
    'Recon-OSScan',
    'Recon-PortScan',
    'Recon-PingSweep',
    'VulnerabilityScan']

CONST_WEB_LABEL = [
    'SqlInjection',
    'BrowserHijacking',
    'CommandInjection',
    'Backdoor_Malware',
    'XSS',
    'Uploading_Attack']

CONST_BRUTE_FORCE_LABEL = [
    'DictionaryBruteForce']

CONST_PROTOCOL_FIELDS = [
    'HTTP',
    'HTTPS',
    'DNS',
    'Telnet',
    'SMTP',
    'SSH',
    'IRC',
    'TCP',
    'UDP',
    'DHCP',
    'ARP',
    'ICMP',
    'IPv',
    'LLC'
]

CONST_DROP_FIELD = [
    'fin_flag_number',
    'syn_flag_number',
    'rst_flag_number',
    'psh_flag_number',
    'ece_flag_number',
    'cwr_flag_number',
    'HTTP',
    'HTTPS',
    'DNS',
    'Telnet',
    'SMTP',
    'SSH',
    'IRC',
    'TCP',
    'UDP',
    'DHCP',
    'ARP',
    'ICMP',
    'IPv',
    'LLC',
    'Min',
    'label'
    ]

CONST_CATEGORICAL_FIELDS = [
    'fin_flag_number',
    'syn_flag_number',
    'rst_flag_number',
    'psh_flag_number',
    'ack_flag_number',
    'ece_flag_number',
    'cwr_flag_number',
    'HTTP',
    'HTTPS',
    'DNS',
    'Telnet',
    'SMTP',
    'SSH',
    'IRC',
    'TCP',
    'UDP',
    'DHCP',
    'ARP',
    'ICMP',
    'IPv',
    'LLC',
]

CONST_TWO_CLASSES_WITH_LSTM = "two_classes_lstm.h5"

CONST_TWO_CLASSES_WITH_RFC = 'two_classes_rfc.pickle'

CONST_EIGHT_CLASSES_WITH_LSTM = 'eight_classes_lstm.h5'