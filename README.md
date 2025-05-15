# CCTV Face Recognition

This is service can recognize faces in real-time from multiple CCTV cameras with an integrated recognition system. This repository provides all the setup and installation instructions to get started with the service.

## Setup MediaMTX and MySQL

### 1. MediaMTX
MediaMTX is a ready-to-use and zero-dependency real-time media server and media proxy that allows to publish, read, proxy, record and playback video and audio streams. It has been conceived as a "media router" that routes media streams from one end to the other.

Follow the official [MediaMTX installation guide](https://github.com/bluenviron/mediamtx?tab=readme-ov-file#installation)

or you can run using this command:
```bash
docker run -d \ 
--name mediamtx-container \
-e MTX_RTSPTRANSPORTS=tcp \
-e MTX_WEBRTCADDITIONALHOSTS=192.168.x.x \
-p 8554:8554 \
-p 1935:1935 \
-p 8888:8888 \
-p 8889:8889 \
-p 8890:8890/udp \
-p 8189:8189/udp \
bluenviron/mediamtx
```

### 2. MySQL

1. **Install MySQL**: Follow the official [MySQL installation guide](https://dev.mysql.com/doc/refman/8.0/en/installing.html).
2. **Create a Database**: Create a MySQL database for this project:
   ```sql
   CREATE DATABASE cctv_db;
   ```
3. **Create a Table**: Create a table in that database:
    ```sql
    CREATE TABLE cctv_logs (
        id SERIAL PRIMARY KEY,
        employee_id VARCHAR(50) NOT NULL,
        name VARCHAR(100) NOT NULL,
        location VARCHAR(100) NOT NULL,
        camera_name VARCHAR(100) NOT NULL,
        status VARCHAR(50) NOT NULL,
        remark TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ```
4. **Configure Database**: Update the connection details in the `config.yaml` file, in `sql_config` section.

## Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/xMaulana/CCTV-FACE-RECOGNITION.git
```

## Create a Virtual Environment (Optional but Recommended)

It is recommended to create a virtual environment to manage your project's dependencies. To do this, follow these steps:

1. **Create a Virtual Environment**:
   Run the following command to create a virtual environment:

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

If you are using conda, you can just create new virtual env using:
```bash
conda create -n env_name
```

## Install Dependencies

1. Install Pytorch using [Pytorch installation guidelines](https://pytorch.org/get-started/locally/)

2. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application using the following command:

```bash
python multiproc.py
```

## Regist face data
To add a new data, create new folder inside `facebank/unprocessed`, the folders that created must follow "ID_NAME" format. Then, insert images into that folder and run the following command:

```bash
python regist_face.py
```