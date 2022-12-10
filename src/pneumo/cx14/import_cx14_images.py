import os
import sys
import urllib.request
from datetime import datetime
import tarfile

ix = os.getcwd().index('umich-mads-capstone-project')
ROOT_PATH = os.path.join(os.getcwd()[:ix], 'umich-mads-capstone-project')
SRC_PATH = os.path.join(ROOT_PATH, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

img_path = os.path.join(ROOT_PATH, 'data/cx14')

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
	  'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
	  'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
	  'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
	  'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
	  'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
	  'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
	  'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]

for idx, link in enumerate(links):
    print(f'timestamp{datetime.now()}')
    fn = os.path.join(img_path,'images_%02d.tar.gz') % (idx+1)
    if not(os.path.isfile(fn)):
      print('Downloading '+fn+'...')
      urllib.request.urlretrieve(link, fn)  # download the zip file
      file = tarfile.open(fn)
      print('Extracting '+fn+'...')
      file.extractall(img_path)
      file.close()
      print('Deleting '+fn+'...')
      os.remove(fn)

print("Download complete. Please check the checksums")