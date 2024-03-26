import asyncio
import logging
import os
import random
import signal
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage
import vocode




logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# gpty=ChatGPTAgent(
#             ChatGPTAgentConfig(
#                 initial_message=BaseMessage(text="How are you doing you?"),
#                 prompt_preamble=f"""The AI is having a pleasant conversation about life and say this number: {random.randint(0, 100)}""",
#             ),
#             openai_api_key=os.getenv("OPENAI_API_KEY"),
#             logger=logger,
#         )
resty= RESTfulUserImplementedAgent(
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
# Old Code
async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False,
        use_blocking_speaker_output=True,
        logger=logger,
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=AssemblyAITranscriber(
            AssemblyAITranscriberConfig.from_input_device(
                microphone_input,
                api_key=os.getenv("ASSEMBLY_AI_API_KEY"),
                
            ),
            logger=logger,
        ),
        agent=resty,
        synthesizer=ElevenLabsSynthesizer(
            ElevenLabsSynthesizerConfig.from_output_device(
                output_device=speaker_output,
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
                stability=1.0,
                similarity_boost=1.0,
            ),
            logger=logger,
        ),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )
    # import pdb; pdb.set_trace();
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())