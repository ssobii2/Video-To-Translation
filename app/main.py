from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from app.routers import translate

app = FastAPI()

# Include the translate router
app.include_router(translate.router)

# Set up templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
