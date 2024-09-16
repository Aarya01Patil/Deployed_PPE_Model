# PPE Detection Model

This project implements a Personal Protective Equipment (PPE) detection model using YOLOv8, deployed with Flask on DigitalOcean.

## Website Preview
![image](https://github.com/user-attachments/assets/c6093167-4405-48e0-8fcd-68b53c34ac24)


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
/
/
├── .do/
│   └── app.yaml
├── app/
│   ├── static/
│   │   ├── css/
│   │   │   └── animated_background.css
│   │   └── images/
│   │       └── PPE_Detection.jpeg
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── result.html
│   ├── __init__.py
│   └── views.py
├── scripts/
│   ├── inference.py
│   └── s3_cleanup.py
├── weights/
│   ├── person_detection/
│   │   └── best.pt
│   └── ppe_detection/
│       └── best.pt
├── requirements.txt
└── run.py
└── setup_ffmpeg.py
└── Procfile
└── Aptfile
└── LICENSE
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

## Deployment

This project is deployed on DigitalOcean using Flask. The `app.py` file contains the Flask application that serves the model predictions.

To run the Flask app locally:

```
python run.py
```

For production deployment, consider using Gunicorn as the WSGI HTTP server:

```
gunicorn run:app
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
