import mimetypes
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import matplotlib.image as mpimg

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))
PATH = Path('data')
train_path = PATH/'TrainVal'

def get_files(c, extensions=None, recurse=False):
    "Return list of files in `c` that have a suffix in `extensions`. `recurse` determines if we search subfolders."
    return [o for o in Path(c).glob('**/*' if recurse else '*')
            if not o.name.startswith('.') and not o.is_dir()
            and (extensions is None or (o.suffix.lower() in extensions))]
def get_image_files(c, check_ext:bool=True, recurse=False):
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=image_extensions, recurse=recurse)

def get_error_imgs(label):
    fnames = get_image_files(train_path/label)
    wrong_img =[]
    for i in fnames:
        try:
            _ = mpimg.imread(i)
        except:
            wrong_img.append(i)
    print(f'Done processing class {label}! {len(wrong_img)} found!')
    f = f'c_{label}.csv'
    if len(wrong_img):
        pd.DataFrame(wrong_img).to_csv(PATH/f)

def main():
    labels = [str(i).split('\\')[-1] for i in train_path.iterdir()]
    # labels = [str(i).split('/')[-1] for i in train_path.ls()] #for linux
    with ProcessPoolExecutor(max_workers=4) as executor:
        for label,_ in zip(labels,executor.map(get_error_imgs,labels)):
            continue
if __name__ == '__main__':
    main()