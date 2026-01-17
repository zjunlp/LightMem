from __future__ import annotations
import os

import base64
import requests
from io import BytesIO
from PIL import Image
import openai

import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from .base import (
    MemoryDataset,
    Trajectory,
    Session,
    QuestionAnswerPair,
    Message,
)


def _parse_session_datetime(dt_str: str) -> datetime:
    """Parse LoCoMo-style datetime string like '1:56 pm on 8 May, 2023'."""
    # Example: "1:56 pm on 8 May, 2023"
    return datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y")

def process_image_to_base64(img_url: str, target_size: tuple = (224, 224)) -> Optional[str]:
    """
    Download image from URL, resize, and convert to base64 encoding.

    Args:
        img_url: Image URL
        target_size: Target image dimensions, defaults to (224, 224)

    Returns:
        Base64 encoded image string, or None if failed
    """
    try:
        # 1. Download image from URL
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()  # Check if request was successful

        # 2. Open image using PIL
        img = Image.open(BytesIO(response.content))

        # 3. Resize image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

        # 4. Convert to RGB mode for consistency
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')

        # 5. Save to memory buffer and convert to base64
        buffered = BytesIO()
        img_resized.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_base64

    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        return None
    except Exception as e:
        print(f"Failed to process image: {e}")
        return None


def analyze_image_with_gpt4o(api_key: str, base_url:str, img_base64: str, prompt: str = "Describe this image") -> Optional[str]:
    """
    Analyze base64 encoded image using GPT-4o.

    Args:
        api_key: OpenAI API key
        img_base64: Base64 encoded image
        prompt: Analysis prompt

    Returns:
        GPT-4o response content, or None if failed
    """
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]

        # Call GPT-4o API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Failed to call GPT-4o API: {e}")
        return None

CATEGORY_ID_TO_TYPE: Dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

class LoCoMo(MemoryDataset):
    """Dataset wrapper for LoCoMo-style long-term multi-session dialogs."""

    @classmethod
    def read_raw_data(cls, path: str) -> LoCoMo:
        """
        - trajectories: List[Trajectory]
        - question_answer_pair_lists: List[List[QuestionAnswerPair]]
        """
        use_gpt4o_caption = os.getenv("USE_GPT4O_CAPTION", "0") == "1"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories: List[Trajectory] = []
        qa_lists: List[List[QuestionAnswerPair]] = []

        image_caption_cache: Dict[str, str] = {}

        for sample_idx, sample in enumerate(data):
            conversation = sample.get("conversation", {})
            speaker_a = conversation.get("speaker_a", "SpeakerA")
            speaker_b = conversation.get("speaker_b", "SpeakerB")

            sessions: List[Session] = []
            session_time_map: Dict[int, datetime] = {}

            for key, value in conversation.items():
                if key.startswith("session_") and key.endswith("_date_time"):
                    # "session_1_date_time" -> "1"
                    parts = key.split("_")
                    if len(parts) < 3:
                        continue
                    try:
                        idx = int(parts[1])
                    except ValueError:
                        continue

                    dt_str = value
                    session_dt = _parse_session_datetime(dt_str)
                    session_time_map[idx] = session_dt

            for s_idx in sorted(session_time_map.keys()):
                msg_key = f"session_{s_idx}"
                raw_msgs = conversation.get(msg_key, [])
                if not raw_msgs:
                    continue

                session_dt = session_time_map[s_idx]
                messages: List[Message] = []

                for msg in raw_msgs:
                    speaker = msg.get("speaker", "")
                    text = msg.get("text", "")

                    msg_metadata: Dict[str, Any] = {
                        "name": speaker,
                    }
                    
                    if speaker == speaker_a:
                        msg_metadata["speaker_tag"] = "speaker_a"
                    elif speaker == speaker_b:
                        msg_metadata["speaker_tag"] = "speaker_b"
                    else:
                        msg_metadata["speaker_tag"] = "unknown"
                    
                    if "dia_id" in msg and msg["dia_id"]:
                        msg_metadata["dia_id"] = msg["dia_id"]
                    
                    img_url = msg.get("img_url")
                    if img_url:
                        msg_metadata["img_url"] = img_url

                        blip_caption = msg.get("blip_caption")
                        if blip_caption:
                            msg_metadata["blip_caption"] = blip_caption
                        
                        if "query" in msg and msg["query"]:
                            msg_metadata["image_query"] = msg["query"]

                        if use_gpt4o_caption:
                            caption: Optional[str] = None
                            if img_url in image_caption_cache:
                                caption = image_caption_cache[img_url]
                            else:
                                img_base64 = process_image_to_base64(img_url)
                                if img_base64:
                                    caption = analyze_image_with_gpt4o(
                                        api_key=os.getenv("OPENAI_API_KEY_FOR_IMAGE", ""),
                                        base_url=os.getenv("OPENAI_API_BASE_FOR_IMAGE", ""),
                                        img_base64=img_base64,
                                        prompt="Please describe the content of this image in detail."
                                    )
                                    if caption:
                                        image_caption_cache[img_url] = caption
                                    else:
                                        caption = "No description available."
                                else:
                                    caption = "No description available."
                            
                            if caption and caption != "No description available.":
                                msg_metadata["gpt4o_image_caption"] = caption

                    role = "user"

                    messages.append(
                        Message(
                            role=role,
                            content=text,
                            timestamp=session_dt,
                            metadata=msg_metadata,
                        )
                    )

                sessions.append(
                    Session(
                        messages=messages,
                        timestamp=session_dt,
                        metadata={
                            "id": f"locomo_session_{sample_idx}_{s_idx}",
                            "speaker_a": speaker_a,
                            "speaker_b": speaker_b,
                        },
                    )
                )

            trajectory = Trajectory(
                sessions=sessions,
                metadata={
                    "id": f"locomo_{sample_idx}",
                    "speaker_a": speaker_a, 
                    "speaker_b": speaker_b,  
                },
            )
            trajectories.append(trajectory)

            if sessions:
                question_ts = sessions[-1].timestamp
            else:
                question_ts = datetime.now()

            qapairs: List[QuestionAnswerPair] = []
            for q_idx, qa in enumerate(sample.get("qa", [])):
                answer = qa.get("answer")
                if answer is None:
                    answer = qa.get("adversarial_answer")

                if isinstance(answer, int):
                    answer = str(answer)

                if answer is not None:
                    answer_list = (answer,)
                else:
                    answer_list = tuple()

                category = qa.get("category")
                category_type = CATEGORY_ID_TO_TYPE.get(category, "unknown")
                metadata: Dict[str, Any] = {
                    "id": f"locomo_q_{sample_idx}_{q_idx}",
                    "question_type": category_type,
                    "category": category_type,
                    "category_id": category,
                    "evidence": qa.get("evidence", []),
                    "speaker_names": [speaker_a, speaker_b],
                }
                if "adversarial_answer" in qa:
                    metadata["adversarial_answer"] = qa["adversarial_answer"]

                qapairs.append(
                    QuestionAnswerPair(
                        role="user",
                        question=qa["question"],
                        answer_list=answer_list,
                        timestamp=question_ts,
                        metadata=metadata,
                    )
                )

            qa_lists.append(qapairs)

        return cls(
            trajectories=trajectories,
            question_answer_pair_lists=qa_lists,
        )

    def _generate_metadata(self) -> Dict[str, Any]:
        dataset_metadata: Dict[str, Any] = {
            "name": "LoCoMo",
            "paper": "Evaluating Very Long-Term Conversational Memory of LLM Agents",
            "paper_url": "https://arxiv.org/abs/2402.17753",
            "codebase_url": "https://github.com/snap-research/LoCoMo",
            "homepage": "https://snap-research.github.io/locomo/",
            "total_sessions": 0,
            "total_messages": 0,
            "total_questions": 0,
            "size": len(self),
        }

        question_type_stats: Dict[str, int] = {}

        for trajectory, qa_list in self:
            dataset_metadata["total_sessions"] += len(trajectory)
            dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(qa_list)

            for qa in qa_list:
                q_type = qa.metadata.get("question_type", "unknown")
                question_type_stats[q_type] = question_type_stats.get(q_type, 0) + 1

        dataset_metadata["question_type_stats"] = question_type_stats

        if len(self) > 0 and dataset_metadata["total_sessions"] > 0:
            dataset_metadata["avg_session_per_trajectory"] = (
                dataset_metadata["total_sessions"] / len(self)
            )
            dataset_metadata["avg_message_per_session"] = (
                dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
            )
            dataset_metadata["avg_question_per_trajectory"] = (
                dataset_metadata["total_questions"] / len(self)
            )
        else:
            dataset_metadata["avg_session_per_trajectory"] = 0.0
            dataset_metadata["avg_message_per_session"] = 0.0
            dataset_metadata["avg_question_per_trajectory"] = 0.0

        return dataset_metadata

    @classmethod
    def filter_questions(self, questions: List[QuestionAnswerPair]) -> List[QuestionAnswerPair]:
        """
        Filter out adversarial questions (category_id=5) for LoCoMo dataset.
        """
        return [qa for qa in questions if qa.metadata.get("category_id") != 5]
    
    @classmethod
    def get_qa_prompt_name(cls, has_graph: bool = False) -> str:
        """Get LoCoMo-specific QA prompt based on whether graph relations exist."""
        if has_graph:
            return "locomo-question-answering-graph-memory-system"
        return "locomo-question-answering-flat-memory-system"
    
    @classmethod
    def get_judge_prompt_info(cls, qa_pair: QuestionAnswerPair) -> Tuple[str, str]:
        """LoCoMo uses a unified judge prompt for all question types."""
        qtype = qa_pair.metadata.get("question_type", "unknown")
        return "locomo-judge", qtype