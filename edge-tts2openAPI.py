from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import edge_tts
import asyncio
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import Optional

# 配置详细日志
from starlette.responses import HTMLResponse

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TTS-Server")

app = FastAPI()


# OpenAI兼容请求模型
class TTSParameters(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    volume: Optional[float] = Field(1.0, ge=0.5, le=3.0)  # 新增音量参数


# 模型到语音配置的映射
MODEL_CONFIG = {
    "tts-1": {
        "quality": "standard",
        "allowed_formats": ["mp3"],
        "voice_map": {
            "alloy": "en-US-GuyNeural",
            "echo": "en-US-JennyNeural",
            "nova": "zh-CN-YunxiNeural"
        }
    },
    "tts-1-hd": {
        "quality": "enhanced",
        "allowed_formats": ["mp3"],
        "voice_map": {
            "alloy": "en-US-AriaNeural",
            "echo": "en-US-DavisNeural",
            "nova": "zh-CN-YunjianNeural"
        }
    }
}


async def write_audio(communicate, stdin):
    """将edge_tts的音频数据写入ffmpeg的stdin"""
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and stdin is not None:
                stdin.write(chunk["data"])
                await stdin.drain()
        if stdin is not None:
            stdin.close()
            await stdin.wait_closed()
    except Exception as e:
        logger.error(f"写入音频失败: {e}")


async def read_audio(stdout):
    """从ffmpeg的stdout读取处理后的音频数据"""
    while True:
        chunk = await stdout.read(4096)
        if not chunk:
            break
        yield chunk


async def generate_edge_audio(text: str, config: dict, voice: str, speed: float, volume: float):
    """根据配置生成音频流"""

    try:
        # 获取真实语音名称
        real_voice = config["voice_map"].get(voice.lower(), voice)

        # 验证语音有效性
        all_voices = await edge_tts.list_voices()
        if not any(v["ShortName"] == real_voice for v in all_voices):
            raise ValueError(f"无效语音: {real_voice}")

        # 根据质量配置调整参数
        rate = "+0%"
        if config["quality"] == "enhanced":
            speed = max(0.8, min(speed, 1.5))

        rate = f"+{int((speed - 1) * 100)}%" if speed != 1.0 else "+0%"

        communicate = edge_tts.Communicate(text, real_voice, rate=rate)

        # 当需要调整音量时使用ffmpeg处理
        if volume != 1.0:
            # 创建ffmpeg进程
            cmd = [
                'ffmpeg',
                '-i', 'pipe:0',
                '-af', f'volume={volume}',
                '-f', 'mp3',
                '-loglevel', 'quiet',
                'pipe:1'
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # 启动写入任务
            writer_task = asyncio.create_task(
                write_audio(communicate, proc.stdin)
            )

            # 直接读取处理后的音频流
            try:
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    yield chunk
            finally:
                # 清理资源
                if proc.stdin:
                    proc.stdin.close()
                await proc.wait()
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

        else:
            # 直接返回原始音频流
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise

@app.post("/v1/audio/speech")
async def create_speech(request: TTSParameters):
    try:
        logger.debug(f"收到请求: {request.dict()}")

        # 检查模型支持
        if request.model not in MODEL_CONFIG:
            raise HTTPException(400, detail=f"不支持的模型: {request.model}")

        config = MODEL_CONFIG[request.model]

        # 验证音频格式
        if request.response_format not in config["allowed_formats"]:
            raise HTTPException(400,
                                detail=f"模型{request.model}不支持格式: {request.response_format}")

        # 获取语音映射
        voice = request.voice.lower()
        if voice not in config["voice_map"]:
            raise HTTPException(400,
                                detail=f"模型{request.model}不支持语音: {request.voice}")

        return StreamingResponse(
            generate_edge_audio(
                text=request.input,
                config=config,
                voice=voice,
                speed=request.speed,
                volume=request.volume
            ),
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "OpenAI-Processing-Ms": "800"
            }
        )

    except HTTPException as he:
        raise
    except Exception as e:
        logger.exception("服务器错误")
        return {
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": 500
            }
        }


# OpenAI兼容的语音列表接口
@app.get("/v1/voices")
async def list_voices():
    return {
        "data": [
            {
                "id": "alloy",
                "name": "Alloy (EdgeTTS)",
                "capacities": ["tts-1", "tts-1-hd"]
            },
            {
                "id": "echo",
                "name": "Echo (EdgeTTS)",
                "capacities": ["tts-1", "tts-1-hd"]
            },
            {
                "id": "nova",
                "name": "Nova (EdgeTTS)",
                "capacities": ["tts-1", "tts-1-hd"]
            }
        ]
    }


@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EdgeTTS 云希语音合成</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 20px auto;
                padding: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
            }}
            .control-group {{
                margin: 15px 0;
            }}
            label {{
                display: block;
                margin-bottom: 5px;
                color: #34495e;
            }}
            textarea {{
                width: 100%;
                height: 120px;
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                resize: vertical;
            }}
            input[type="range"] {{
                width: 100%;
            }}
            button {{
                background: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.3s;
            }}
            button:hover {{
                background: #2980b9;
            }}
            #audioPlayer {{
                margin-top: 20px;
                width: 100%;
            }}
            .speed-controls {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>EdgeTTS 云希语音合成</h1>

            <div class="control-group">
                <label for="textInput">输入要合成的文本：</label>
                <textarea id="textInput" placeholder="请输入中文文本..."></textarea>
            </div>

            <div class="speed-controls">
                <div class="control-group">
                    <label for="speed">语速 (0.5-2.0): <span id="speedValue">1.0</span></label>
                    <input type="range" id="speed" min="0.5" max="2.0" step="0.1" value="1.0">
                </div>

                <div class="control-group">
                    <label for="volume">音量 (0.5-3.0): <span id="volumeValue">1.0</span></label>
                    <input type="range" id="volume" min="0.5" max="3.0" step="0.1" value="1.0">
                </div>
            </div>

            <button onclick="generateSpeech()">生成并播放语音</button>
            <audio id="audioPlayer" controls></audio>
        </div>

        <script>
            function updateSpeedValue() {{
                document.getElementById('speedValue').textContent = document.getElementById('speed').value;
            }}

            function updateVolumeValue() {{
                document.getElementById('volumeValue').textContent = document.getElementById('volume').value;
            }}

            document.getElementById('speed').addEventListener('input', updateSpeedValue);
            document.getElementById('volume').addEventListener('input', updateVolumeValue);

            async function generateSpeech() {{
                const text = document.getElementById('textInput').value;
                const speed = document.getElementById('speed').value;
                const volume = document.getElementById('volume').value;
                const audioPlayer = document.getElementById('audioPlayer');

                if (!text) {{
                    alert('请输入要合成的文本');
                    return;
                }}

                try {{
                    const response = await fetch('/v1/audio/speech', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            input: text,
                            model: "tts-1",
                            voice: "nova",
                            speed: parseFloat(speed),
                            volume: parseFloat(volume)
                        }})
                    }});

                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}

                    const audioBlob = await response.blob();
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayer.src = audioUrl;
                    audioPlayer.play();
                }} catch (error) {{
                    console.error('Error:', error);
                    alert('生成语音失败: ' + error.message);
                }}
            }}
        </script>
    </body>
    </html>
    """
if __name__ == "__main__":
    logger.info("启动TTS服务 (端口 13241)...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=13241,
        log_config=None,
        timeout_keep_alive=600
    )
