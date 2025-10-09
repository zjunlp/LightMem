from __future__ import annotations
import json 
from .base import (
    MemoryDataset, 
    Trajectory, 
    Session, 
    QuestionAnswerPair, 
    Message, 
)
from datetime import datetime
from typing import Dict, Any 


class LongMemEval(MemoryDataset):

    @classmethod
    def read_raw_data(cls, path: str) -> LongMemEval:
        with open(path, 'r') as f:
            data = json.load(f)

        trajectories, question_answer_pair_lists = [], [] 
        for sample in data:
            question_datetime = datetime.strptime(sample["question_date"], '%Y/%m/%d (%a) %H:%M')
            question_type = sample["question_type"]
            answer = sample["answer"]
            # Some answers are integers, which should be converted to strings. 
            if isinstance(answer, int): 
                answer = str(answer)
            question_answer_pair = QuestionAnswerPair(
                role="user",
                question=sample["question"],
                answer_list=(answer, ), 
                timestamp=question_datetime, 
                metadata={
                    "question_type": question_type,
                    "id": sample["question_id"],
                    "answer_session_ids": sample["answer_session_ids"],
                }
            )

            num_sessions = len(sample["haystack_sessions"])
            trajectory = [] 
            for i in range(num_sessions):
                raw_session = sample["haystack_sessions"][i]
                # There are some empty sessions to be skipped. 
                if len(raw_session) == 0:
                    continue
                session_id = sample["haystack_session_ids"][i]
                session_date = sample["haystack_dates"][i]
                session_datetime = datetime.strptime(session_date, '%Y/%m/%d (%a) %H:%M')
                session = Session(
                    messages=[
                        Message(
                            role=message["role"],
                            content=message["content"],
                            timestamp=session_datetime,
                        )
                        for message in raw_session
                    ],
                    timestamp=session_datetime,
                    metadata={
                        "id": session_id,
                    }
                )
                trajectory.append(session)

            trajectories.append(
                Trajectory(
                    sessions=trajectory,
                    metadata={
                        "id": f"longmemeval_{sample['question_id']}",
                    }
                )
            )
            question_answer_pair_lists.append([question_answer_pair])

        return cls(
            trajectories=trajectories,
            question_answer_pair_lists=question_answer_pair_lists
        )

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate the metadata of the dataset."""
        dataset_metadata = {
            "name": "LongMemEval",
            "paper": "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory", 
            "codebase_url": "https://github.com/xiaowu0162/LongMemEval", 
            "total_sessions": 0, 
            "total_messages": 0, 
            "total_questions": 0, 
            "size": len(self)
        } 
        question_type_stats = {}  

        for trajectory, question_answer_pair_list in self: 
            dataset_metadata["total_sessions"] += len(trajectory)
            dataset_metadata["total_messages"] += sum(len(session) for session in trajectory)
            dataset_metadata["total_questions"] += len(question_answer_pair_list)
            for question_answer_pair in question_answer_pair_list: 
                question_type = question_answer_pair.metadata["question_type"]
                question_type_stats[question_type] = question_type_stats.get(question_type, 0) + 1

        dataset_metadata["question_type_stats"] = question_type_stats
        dataset_metadata["avg_session_per_trajectory"] = dataset_metadata["total_sessions"] / len(self)
        dataset_metadata["avg_message_per_session"] = dataset_metadata["total_messages"] / dataset_metadata["total_sessions"]
        dataset_metadata["avg_question_per_trajectory"] = dataset_metadata["total_questions"] / len(self)

        return dataset_metadata
