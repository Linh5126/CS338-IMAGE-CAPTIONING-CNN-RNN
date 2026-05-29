import argparse
from utils import create_input_files

if __name__ == '__main__':
    # Khởi tạo bộ đọc tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description='Tạo file HDF5 và JSON cho đồ án Image Captioning')
    parser.add_argument('--dataset', type=str, required=True, choices=['flickr8k', 'flickr32k'], 
                        help='Chọn bộ dữ liệu muốn nén: flickr8k hoặc flickr32k')
    
    args = parser.parse_args()

    # Tự động chọn đường dẫn dựa trên lệnh bạn gõ
    if args.dataset == 'flickr8k':
        json_path = '/content/drive/MyDrive/CS338/dataset_flickr8k.json'
        img_folder = '/content/drive/MyDrive/CS338/flickr8k/Images/'
    elif args.dataset == 'flickr30k':
        json_path = '/content/drive/MyDrive/CS338/dataset_flickr32k.json'
        img_folder = '/content/drive/MyDrive/CS338/flickr32k/Images/'

    print(f"🚀 Bắt đầu xử lý bộ dữ liệu: {args.dataset.upper()}...")
    
    # Thực thi hàm tạo file
    create_input_files(dataset=args.dataset,
                       karpathy_json_path=json_path,
                       image_folder=img_folder,
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/content/drive/MyDrive/CS338/',
                       max_len=50)
                       
    print(f"✅ Đã xử lý xong {args.dataset.upper()}! Các file HDF5 và JSON đã được lưu vào Drive.")