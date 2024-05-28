
import gdown 


def main():
    url = "https://drive.google.com/drive/folders/1MzkXMf6I_S7Jr9V-ntfjjYA_ag7Zn0y5?usp=sharing"

    gdown.download_folder(url, quiet=True)
    
    
if __name__ == '__main__':
    main()