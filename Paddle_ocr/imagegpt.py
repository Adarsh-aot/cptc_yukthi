from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import uvicorn
from paddleocr import PaddleOCR,draw_ocr
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()


origins = [
    "http://127.0.0.1:9000",
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        with open(file.filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get the absolute path of the saved file
        saved_file_path = os.path.abspath(file.filename)
        l = []
        
        img_path = saved_file_path
        result = ocr.ocr(img_path, cls=True)
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                print(line)
                l.append(line[1][0])
        
        k = ''
        for i in l:
            k = k + ' ' + i
        

        os.remove(saved_file_path)
        return JSONResponse(status_code=200, content={"message": saved_file_path, "Result": k})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error occurred: {str(e)}"})



# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=9000)