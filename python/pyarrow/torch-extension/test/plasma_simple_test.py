# plasma_store -m 1000000000 -s /tmp/plasma &
import pyarrow.plasma as plasma
client = plasma.connect("/tmp/plasma")
teststring = 'Sample Bytes, Oh yeah!'
print("Test Input:", teststring)
a = bytearray(teststring, 'utf-8')
obj_id = client.put_raw_buffer(a, device_num=1)
print("object id:", obj_id)
res = client.get(obj_id, device_num=1)
print("we got:", res, "its value is:", res.to_pybytes())

