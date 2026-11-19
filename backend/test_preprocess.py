from preprocess import preprocess_file


file_path = "sample.jpg"

try:
    file_type, tensor = preprocess_file(file_path)

    print("File type:", file_type)
    print("Tensor shape:", tensor.shape)

except Exception as error:
    print("Error:", error)