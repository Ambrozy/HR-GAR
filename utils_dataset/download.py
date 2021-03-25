import os
import requests
import zipfile
import tarfile


def download_url(url, save_path, chunk_size=128):
    print('Start downloading', save_path)
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('Save', save_path)


def extract_files_in_folder(directory):
    zip_files = os.listdir(directory)
    for name in zip_files:
        output_name = os.path.splitext(name)[0]
        opener, mode = None, None
        if name.endswith('.zip'):
            opener, mode = zipfile.ZipFile, 'r'
        elif name.endswith('.tgz'):
            opener, mode = tarfile.open, 'r:gz'

        if opener and mode:
            with opener(LOAD_DIR + name, mode) as zip_ref:
                zip_ref.extractall(path=LOAD_DIR + output_name)
            print('Archive', name, 'unpacked to', output_name)
        else:
            print('Unknown archive format', name)


if __name__ == '__main__':
    LOAD_DIV2K_URL_PATH = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/'
    LOAD_DIR = '../DIV2K/'
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_train_HR.zip', LOAD_DIR + 'DIV2K_train_HR.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_valid_HR.zip', LOAD_DIR + 'DIV2K_valid_HR.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_train_LR_x8.zip', LOAD_DIR + 'DIV2K_train_LR_x8.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_train_LR_mild.zip', LOAD_DIR + 'DIV2K_train_LR_mild.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_train_LR_difficult.zip', LOAD_DIR + 'DIV2K_train_LR_difficult.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_train_LR_wild.zip', LOAD_DIR + 'DIV2K_train_LR_wild.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_valid_LR_x8.zip', LOAD_DIR + 'DIV2K_valid_LR_x8.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_valid_LR_mild.zip', LOAD_DIR + 'DIV2K_valid_LR_mild.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_valid_LR_difficult.zip', LOAD_DIR + 'DIV2K_valid_LR_difficult.zip')
    download_url(LOAD_DIV2K_URL_PATH + 'DIV2K_valid_LR_wild.zip', LOAD_DIR + 'DIV2K_valid_LR_wild.zip')
    extract_files_in_folder(LOAD_DIR)


    LOAD_DIV2K_URL_PATH = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/'
    LOAD_DIR = '../Flowers/'
    download_url(LOAD_DIV2K_URL_PATH + '17/17flowers.tgz', LOAD_DIR + '17flowers.tgz')
    download_url(LOAD_DIV2K_URL_PATH + '102/102flowers.tgz', LOAD_DIR + '102flowers.tgz')
    extract_files_in_folder(LOAD_DIR)
