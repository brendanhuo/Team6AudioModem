import bitarray
import binascii

# convert a string into binary using utf-8 encoding
def toBinary(string):
    # Text to Binary
    return '0' + bin(int.from_bytes(string.encode(), 'big'))[2:]

# convert a text file into a string of binary 
def fileToBinary(filename):
    file = open(filename)
    line = file.read().replace("\n", " ")
    file.close()
    binary = '0' + bin(int.from_bytes(line.encode(), 'big'))[2:]
    return binary

# return a string of binary from a text file of binary information
def binaryTextFileToBinary(filename):
    file = open(filename)
    line = file.read().replace("\n", " ")
    file.close()
    return line

# Binary string to Bytes
def str_to_bytearray(string_data):
    new_data = []
    for i in range(0, len(string_data), 8):
        new_data.append(string_data[i: i+8])  

    int_data = [] 
    for i in new_data:
        int_data.append(int(i, 2))

    return bytearray(int_data)

def text_from_bits(bits, encoding='utf-8', errors='ignore'):
    """Convert byte sequence to text"""
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    """Converts bit stream to bytes"""
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

def calculateBER(ba, audioOutput):
    """Calculates bit error rate"""
    errorCount = 0
    for i in range(len(audioOutput)):
        if ba[i] != audioOutput[i]:
            errorCount += 1
    return errorCount / len(ba)
