# PPE Detection Model

This project implements a Personal Protective Equipment (PPE) detection model using YOLOv8, deployed with Flask on DigitalOcean.

## Project Overview

The PPE detection system consists of two main components:
1. A person detection model trained on full images
2. A PPE detection model trained on cropped person images

The system processes images through both models to identify persons and their PPE equipment.

## Dataset

The dataset used for training contains images and annotations for the following classes:
- Person
- Hard-hat
- Gloves
- Mask
- Glasses
- Boots
- Vest
- PPE-suit
- Ear-protector
- Safety-harness

## Project Structure

```
project/
│
├── Datasets.zip
│   ├── images/
│   ├── annotations/
│   └── classes.txt
│
├── scripts/
│   ├── pascalVOC_to_yolo.py
│   └── inference.py
│
├── weights/
│   ├── person_detection.pt
│   └── ppe_detection.pt
│
├── app.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/ppe-detection.git
   cd ppe-detection
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

To convert PascalVOC annotations to YOLOv8 format:

```
python scripts/pascalVOC_to_yolo.py --input_dir /path/to/input --output_dir /path/to/output
```

### Model Training

Train the person detection and PPE detection models using the YOLOv8 framework. Refer to the [YOLOv8 documentation](https://docs.ultralytics.com/) for detailed instructions.

### Inference

To run inference on a directory of images:

```
python scripts/inference.py --input_dir /path/to/input_images --output_dir /path/to/output_images --person_det_model weights/person_detection.pt --ppe_detection_model weights/ppe_detection.pt
```

## Deployment

This project is deployed on DigitalOcean using Flask. The `app.py` file contains the Flask application that serves the model predictions.

To run the Flask app locally:

```
python app.py
```

For production deployment, consider using Gunicorn as the WSGI HTTP server:

```
gunicorn app:app
```

## API Endpoints

- `/predict`: POST endpoint that accepts an image file and returns the PPE detection results.

Example usage:
```python
import requests

url = "http://your-digitalocean-ip/predict"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Performance and Evaluation

Refer to the project report for detailed information on:
- Model architecture choices
- Training approaches
- Evaluation metrics
- Performance optimizations

## Future Improvements

- Implement real-time video processing
- Add a web interface for easy image upload and visualization
- Expand the model to detect additional PPE classes

## Contributing

Contributions to improve the project are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Create a new Pull Request

## License

This project is licensed under the Personal License. See the [LICENSE](LICENSE) file for details.

## Contact

For any queries or support, please open an issue in the GitHub repository.
