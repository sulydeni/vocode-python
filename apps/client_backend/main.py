import logging
import os
from fastapi import FastAPI

from vocode.streaming.agent.restful_user_implemented_agent import RESTfulUserImplementedAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig, RESTfulUserImplementedAgentConfig
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig, ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import AssemblyAITranscriberConfig
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from customConversation import ConversationRouter
from vocode.streaming.models.message import BaseMessage

from dotenv import load_dotenv

from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.transcriber.assembly_ai_transcriber import AssemblyAITranscriber

load_dotenv()

app = FastAPI()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

rest_lifex_call= RESTfulUserImplementedAgent(
    RESTfulUserImplementedAgentConfig(
        initial_message=BaseMessage(text="How are you doing you?"),
        respond=RESTfulUserImplementedAgentConfig.EndpointConfig(
            url="https://api-v1.yourlifex.com/api/v1/response_api/fd69a44d-d0e9-45f9-a948-30c270e7bacc/",
            method="POST"
        ),
        generate_responses=False,
    ),
    logger=logger
    )

conversation_router = ConversationRouter(
    agent_thunk=lambda: rest_lifex_call,
    transcriber_thunk= lambda input_audio_config: AssemblyAITranscriber(
            AssemblyAITranscriberConfig.from_input_audio_config(
                input_audio_config=input_audio_config,
                api_key=os.getenv("ASSEMBLY_AI_API_KEY"),
            )
        ),
        synthesizer_thunk= lambda output_audio_config: ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_audio_config(
                output_audio_config=output_audio_config,
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
                stability=1.0,
                similarity_boost=1.0,
            ),
        ),
    logger=logger,
)

app.include_router(conversation_router.get_router())
