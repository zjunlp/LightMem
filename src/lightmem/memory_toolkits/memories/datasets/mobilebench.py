from __future__ import annotations
import json 
from .base import (
    MemoryDataset, 
    Trajectory, 
    Session, 
    QuestionAnswerPair, 
    Message, 
)
from datetime import datetime, timedelta
from typing import Dict, Any 


class MobileBench(MemoryDataset):

    @classmethod
    def read_raw_data(cls, path: str) -> MobileBench:
        with open(path, 'r') as f:
            data = json.load(f)

        trajectories, question_answer_pair_lists = [], [] 
        for i, sample in enumerate(data):
            trajectory = sample["sessions"]
            formatted_trajectory = []
            for session in trajectory: 
                messages = session["messages"]
                session_meta_data = {
                    key: value
                    for key, value in session.items() if key != "messages"
                }
                session_timestamp = datetime.strptime(session["ended_at"], '%Y-%m-%d %H:%M:%S')

                formatted_messages = []
                for message in messages:
                    message_timestamp = datetime.fromisoformat(message["timestamp"])
                    formatted_messages.append(
                        Message(
                            role=message["role"],
                            content=message["content"],
                            timestamp=message_timestamp,
                            metadata={
                                "name": message["name"],
                                "side_note": message["side_note"],
                            }
                        )
                    )

                formatted_session = Session(
                    messages=formatted_messages,
                    timestamp=session_timestamp,
                    metadata=session_meta_data,
                )
                formatted_trajectory.append(formatted_session)
            
            # last_session_timestamp = formatted_trajectory[-1].timestamp
            last_session_timestamp = session_timestamp
            formatted_trajectory = Trajectory(
                sessions=formatted_trajectory,
                metadata={
                    "id": f"mobilebench_{i}",
                }
            )
            trajectories.append(formatted_trajectory)

            formatted_qa_pairs = [] 
            tool_book = sample["question_type_toolbook"] 
            for qtype in tool_book["question_types"]:
                for qa_pair in qtype["qa_pairs"]:
                    question_timestamp = last_session_timestamp + timedelta(days=7.0)
                    formatted_qa_pairs.append(
                        QuestionAnswerPair(
                            role="user",
                            question=qa_pair["question"],
                            answer_list=tuple(qa_pair["golden_answers"]), 
                            timestamp=question_timestamp, 
                            metadata={
                                "question_type": qa_pair["question_type"],
                                "id": qa_pair["id"],
                                "difficulty": qa_pair["difficulty"],
                                "question_form": qa_pair["question_form"],
                                "source_evidences": qa_pair["source_evidences"],
                                "topic": qa_pair["topic"],
                                "num_hops": qa_pair["num_hops"],
                                "num_sub_questions": qa_pair["num_sub_questions"],
                                "effective_timestamp": qa_pair["effective_timestamp"],
                                "side_note": qa_pair["side_note"],
                                "is_consumed": qa_pair["is_consumed"],
                            }
                        )
                    )
            question_answer_pair_lists.append(formatted_qa_pairs)

        return cls(
            trajectories=trajectories,
            question_answer_pair_lists=question_answer_pair_lists
        )

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate the metadata of the dataset."""
        dataset_metadata = {
            "name": "MobileBench",
            "total_sessions": 0, 
            "total_messages": 0, 
            "total_questions": 0, 
            "size": len(self)
        } 
        question_stats = {
            "question_type": {},
            "difficulty": {},
            "question_form": {},
            "num_hops": {},
        }  

        for trajectory, question_answer_pair_list in self: 
            dataset_metadata["total_sessions"] += len(trajectory)
            dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(question_answer_pair_list)
            for question_answer_pair in question_answer_pair_list: 
                question_type = question_answer_pair.metadata["question_type"]
                question_stats["question_type"][question_type] = question_stats["question_type"].get(question_type, 0) + 1
                difficulty = question_answer_pair.metadata["difficulty"]
                question_stats["difficulty"][difficulty] = question_stats["difficulty"].get(difficulty, 0) + 1
                form = question_answer_pair.metadata["question_form"]
                question_stats["question_form"][form] = question_stats["question_form"].get(form, 0) + 1
                num_hops = str(question_answer_pair.metadata["num_hops"])
                question_stats["num_hops"][num_hops] = question_stats["num_hops"].get(num_hops, 0) + 1

        dataset_metadata["question_stats"] = question_stats
        dataset_metadata["avg_session_per_trajectory"] = dataset_metadata["total_sessions"] / len(self)
        dataset_metadata["avg_message_per_session"] = dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
        dataset_metadata["avg_question_per_trajectory"] = dataset_metadata["total_questions"] / len(self)

        return dataset_metadata
