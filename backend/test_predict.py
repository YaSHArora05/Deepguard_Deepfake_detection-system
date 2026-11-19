from predict import predict_file


file_path = "sample.jpg"

try:
    result = predict_file(file_path)

    print(result)

except Exception as error:
    print("Error:", error)