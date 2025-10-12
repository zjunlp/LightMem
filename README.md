<h1 align="center"> LightMem </h1>

<p align="center">
  <a href="https://arxiv.org/abs/xxxxx">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv">
  </a>
  <a href="https://github.com/zjunlp/LightMem">
    <img src="https://img.shields.io/github/stars/zjunlp/LightMem?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/zjunlp/LightMem/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
  <img src="https://img.shields.io/github/last-commit/zjunlp/LightMem?color=blue" alt="Last Commit">
  <img src="https://img.shields.io/badge/PRs-Welcome-red" alt="PRs Welcome">
</p>

<h5 align="center"> ⭐ If you like our project, please give us a star on GitHub for the latest updates!</h5>

---

**LightMem** is a lightweight and efficient memory management framework designed for Large Language Models and AI Agents. It provides a simple yet powerful memory storage, retrieval, and update mechanism to help you quickly build intelligent applications with long-term memory capabilities.

- **Paper**: (coming soon)
- **GitHub**: https://github.com/zjunlp/LightMem

<span id='features'/>

## ✨ Key Features

* 🚀 **Lightweight & Efficient**
  <br> Minimalist design with minimal resource consumption and fast response times

* 🎯 **Easy to Use**
  <br> Simple API design - integrate into your application with just a few lines of code

* 🔌 **Flexible & Extensible**
  <br> Modular architecture supporting custom storage engines and retrieval strategies

* 🌐 **Broad Compatibility**
  <br> Support for mainstream LLMs (OpenAI, Qwen, DeepSeek, etc.)

<span id='news'/>

## 📢 News

- **[2025-10-12]**: 🎉 LightMem project is officially Open-Sourced!

<span id='todo'/>

## ☑️ Todo List
LightMem is continuously evolving! Here's what's coming:
    
- Offline Pre-computation of KV Cache for Update (Lossless)
- Online Pre-computation of KV Cache Before Q&A (Lossy)
- MCP (Memory Control Policy)
- Integration of Common Models and Feature Enhancement
- Coordinated Use of Context and Long-Term Memory Storage


<span id='contents'/>

## 📑 Table of Contents

* <a href='#features'>✨ Key Features</a>
* <a href='#news'>📢 News</a>
* <a href='#todo'>☑️ Todo List</a>
* <a href='#installation'>🔧 Installation</a>
* <a href='#quickstart'>⚡ Quick Start</a>
* <a href='#architecture'>🏗️ Architecture</a>
* <a href='#examples'>💡 Examples</a>
* <a href='#configuration'>⚙️ Configuration</a>
* <a href='#contributors'>👥 Contributors</a>

<span id='installation'/>

## 🔧 Installation

### Installation Steps

#### Option 1: Install from Source 
```bash
# Clone the repository
git clone https://github.com/zjunlp/LightMem.git
cd LightMem

# Create virtual environment
conda create -n lightmem python=3.10 -y
conda activate lightmem

# Install dependencies
unset ALL_PROXY
pip install -e .
```

#### Option 2: Install via pip
```bash
pip install lightmem  # Coming soon
```

## ⚡ Quick Start
```python
cd experiments
python run_lightmem_qwen.py
```

<span id='architecture'/>

## 🏗️ Architecture

### 🗺️ Core Modules Overview
LightMem adopts a modular design, breaking down the memory management process into several pluggable components. The core directory structure exposed to users is outlined below, allowing for easy customization and extension:

```python
LightMem/
├── src/lightmem/            # Main package
│   ├── __init__.py          # Package initialization
│   ├── configs/             # Configuration files
│   ├── factory/             # Factory methods
│   ├── memory/              # Core memory management
│   └── memory_toolkits/     # Memory toolkits
├── experiments/             # Experiment scripts
├── datasets/                # Datasets files
└── examples/                # Examples
```

### 🧩 Supported Backends per Module

The following table lists the backends values currently recognized by each configuration module. Use the `model_name` field (or the corresponding config object) to select one of these backends.

| Module (config)                 | Supported backends |
| :---                            | :--- |
| `PreCompressorConfig`           | `llmlingua-2`, `entropy_compress` |
| `TopicSegmenterConfig`          | `llmlingua-2` |
| `MemoryManagerConfig`           | `openai`, `deepseek` |
| `TextEmbedderConfig`            | `huggingface` |
| `MMEmbedderConfig`              | `huggingface` |
| `EmbeddingRetrieverConfig`      | `qdrant` |

<span id='examples'/>

## 💡 Examples

### Initialize LightMem
```python
import os
from lightmem.memory.lightmem import LightMemory
from lightmem.configs.base import BaseMemoryConfigs
API_KEY='YOUR_API_KEY'
API_BASE_URL=''
LLM_MODEL=''
EMBEDDING_MODEL_PATH='/your/path/to/models/all-MiniLM-L6-v2'
LLMLINGUA_MODEL_PATH='/your/path/to/models/llmlingua-2-bert-base-multilingual-cased-meetingbank'
config_dict = {
    "pre_compress": True,
    "pre_compressor": {
        "model_name": "llmlingua-2",
        "configs": {
            "llmlingua_config": {
                "model_name": LLMLINGUA_MODEL_PATH,
                "device_map": "cuda",
                "use_llmlingua2": True,
            },
        }
    },
    "topic_segment": True,
    "precomp_topic_shared": True,
    "topic_segmenter": {
        "model_name": "llmlingua-2",
    },
    "messages_use": "user_only",
    "metadata_generate": True,
    "text_summary": True,
    "memory_manager": {
        "model_name": "openai",
        "configs": {
            "model": LLM_MODEL,
            "api_key": API_KEY,
            "max_tokens": 16000,
            "openai_base_url": API_BASE_URL
        }
    },
    "extract_threshold": 0.1,
    "index_strategy": "embedding",
    "text_embedder": {
        "model_name": "huggingface",
        "configs": {
            "model": EMBEDDING_MODEL_PATH,
            "embedding_dims": 384,
            "model_kwargs": {"device": "cuda"},
        },
    },
    "retrieve_strategy": "embedding",
    "embedding_retriever": {
        "model_name": "qdrant",
        "configs": {
            "collection_name": "my_long_term_chat",
            "embedding_model_dims": 384,
            "path": "./my_long_term_chat", 
        }
    },
    "update": "offline",
}
lightmem = LightMemory.from_config(config_dict)
```

### Add Memory
```python

### Add Memory
session = {
"timestamp": "2025-01-10",
"turns": [
    [
        {"role": "user", "content": "My favorite ice cream flavor is pistachio, and my dog's name is Rex."}, 
        {"role": "assistant", "content": "Got it. Pistachio is a great choice."}, 
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"}, 
        {"role": "assistant", "content": "How about watching 'Inception'? It's a great movie!"},
        {"role": "user", "content": "I'm trying to stay on top of my fitness goals and was wondering if you could recommend some workouts that can help me increase my step count. By the way, I've been tracking my progress with my new Fitbit Inspire HR, which I bought on February 15th - it's been really motivating me to move more!"}, 
        {"role": "assistant", "content": "Congratulations on taking the first step (pun intended!) towards a healthier lifestyle! I'm happy to help you with workout recommendations to increase your step count.\n\nFirstly, kudos on investing in a Fitbit Inspire HR! It's great that you've been tracking your progress and finding motivation from it.\n\nNow, let's get moving! Here are some effective workouts to help you boost your step count:\n\n1. **Brisk Walking**: Aim for a 30-minute brisk walk, 3-4 times a week. This will not only increase your step count but also improve cardiovascular health and burn calories.\n2. **High-Intensity Interval Training (HIIT)**: Incorporate HIIT workouts that involve short bursts of intense exercise followed by brief rest periods. This type of exercise can help you take more steps in less time. For example, try 30 seconds of jogging or jumping jacks followed by 30 seconds of rest.\n3. **Stair Climbing**: Find a staircase or a stairmaster at your local gym and climb for 10-15 minutes, 2-3 times a week. This will not only increase your step count but also strengthen your legs and glutes.\n4. **Bodyweight Exercises**: Incorporate bodyweight exercises like squats, lunges, and calf raises into your daily routine. These exercises can be done in short intervals throughout the day to increase your step count.\n5. **Walking Lunges**: Perform walking lunges while holding dumbbells or without weights. This exercise targets your legs and glutes, and you can do it anywhere, anytime.\n6. **Jumping Rope**: Jumping rope is a great cardio exercise that can be done in short intervals to increase your step count. Aim for 3-5 minutes of jumping rope, 2-3 times a week.\n7. **Dance-Based Workouts**: Try dance-based workouts like Zumba or Hip Hop Abs that involve high-energy movements to increase your step count and have fun while doing it!\n\nRemember to always listen to your body and start slowly, especially if you're new to exercise. It's essential to pace yourself and gradually increase the intensity and duration of your workouts.\n\nAdditionally, here are some tips to help you increase your step count throughout the day:\n\n* Take the stairs instead of the elevator at work or in your home.\n* Park your car farther away from your destination to get in a short walk.\n* Take a short walk during your lunch break or after dinner.\n* Do a few laps around your office or house during commercial breaks while watching TV.\n\nI hope these workouts and tips help you reach your fitness goals and stay motivated with your Fitbit Inspire HR!"}, 
        {"role": "user", "content": "I've been doing some yoga in the morning, and I'm curious to know if there are any specific yoga poses that can help improve my sleep quality."}, 
        {"role": "assistant", "content": "Yoga is an excellent way to start the day! Yes, certain yoga poses can indeed help improve sleep quality. Here are some sleep-promoting yoga poses you can incorporate into your morning routine:\n\n1. **Child's Pose (Balasana)**: This gentle, relaxing pose can help calm your mind and body, making it easier to fall asleep. Kneel on your mat, then sit back onto your heels. Stretch your arms out in front, lower your forehead to the ground, and breathe deeply.\n2. **Downward-Facing Dog (Adho Mukha Svanasana)**: This pose can help relax your neck, shoulders, and entire back, which can contribute to better sleep. Start on all fours, then lift your hips up and back, straightening your arms and legs. Hold for 5-10 breaths.\n3. **Pigeon Pose (Eka Pada Rajakapotasana)**: This pose can help release tension in your hips and lower back, which can improve sleep quality. Start on all fours, then bring one knee forward and place your ankle in front of the other knee. Lower your torso down, stretching the back leg. Hold for 5-10 breaths on each side.\n4. **Seated Forward Fold (Paschimottanasana)**: This pose can help calm your nervous system and promote relaxation. Sit on the floor with your legs straight out in front of you. Inhale, lengthen your spine, and then exhale, folding forward, reaching for your toes or shins. Hold for 5-10 breaths.\n5. **Legs Up The Wall Pose (Viparita Karani)**: This inverted pose can help calm your nervous system, reduce swelling in the legs, and promote relaxation. Lie on your back with your legs straight up against a wall. Stay for 5-10 minutes, breathing deeply.\n6. **Savasana (Corpse Pose)**: This pose is often referred to as the \"relaxation pose.\" It can help quiet your mind and body, preparing you for a restful night's sleep. Lie on your back, arms and legs relaxed, eyes closed, and focus on your breath. Stay for 5-10 minutes.\n7. **Cat-Cow Pose (Marjaryasana-Bitilasana)**: This gentle flow can help relax your spine, neck, and entire body, making it easier to fall asleep. Start on all fours, inhaling and arching your back (cat pose). Exhale, rounding your back and tucking your chin to your chest (cow pose). Repeat for 5-10 breaths.\n\nIn addition to these poses, consider the following tips to enhance your yoga practice for better sleep:\n\n* Practice yoga at a gentle, relaxing pace to promote calmness and reduce stimulating energy.\n* Avoid vigorous or energizing yoga practices, such as hot yoga or dynamic flows, close to bedtime.\n* Focus on deep, slow breathing throughout your practice to calm your nervous system.\n* End your practice with a relaxing Savasana or meditation to quiet your mind and body.\n\nRemember to listen to your body and modify or avoid any poses that cause discomfort or pain. Happy practicing!"}, 
        {"role": "user", "content": "That's really helpful, thanks! By the way, I've also been using a foam roller for post-workout stretching, and it's made a huge difference in reducing muscle soreness. I got it from Amazon, and it arrived on March 2nd. Anyway, I've been trying to use it at least three times a week, usually after my morning yoga sessions."}, 
        {"role": "assistant", "content": "Foam rolling is an excellent addition to your fitness routine! It's great to hear that it's been helping with reducing muscle soreness. Consistency is key, so it's awesome that you're aiming to use it at least three times a week.\n\nUsing a foam roller after your morning yoga sessions is a great way to enhance your stretching routine. It can help:\n\n1. **Reduce muscle soreness**: As you've already experienced, foam rolling can help reduce muscle soreness and inflammation.\n2. **Improve circulation**: Foam rolling can help increase blood flow to the affected areas, which can aid in the removal of waste products and promote healing.\n3. **Break up adhesions**: Foam rolling can help break up adhesions and knots in the muscles, which can improve range of motion and reduce stiffness.\n4. **Enhance recovery**: Foam rolling can help speed up the recovery process by reducing muscle spasms and promoting relaxation.\n\nRemember to focus on areas that feel tight or sore, and don't be afraid to experiment with different rolling techniques and pressures to find what works best for you.\n\nAs you continue to use your foam roller, keep in mind the following tips:\n\n* Start with gentle pressure and gradually increase as needed.\n* Roll slowly and control the movement to target specific areas.\n* Breathe deeply and relax while rolling to get the most benefit.\n* Don't roll over bony areas or joints.\n* Clean and maintain your foam roller regularly to prevent bacterial growth.\n\nIt's great that you're taking proactive steps to prioritize your fitness and recovery. Keep up the good work, and don't hesitate to reach out if you have any more questions or need further guidance!"}, 
        {"role": "user", "content": "I've also been tracking my blood pressure regularly with my new wireless blood pressure monitor from Omron, which I got on March 10th. I've been trying to keep an eye on it since my last check-up showed slightly higher than normal readings. Do you have any tips on how to lower blood pressure naturally?"}, 
        {"role": "assistant", "content": "Monitoring your blood pressure regularly is a great step towards taking control of your health! Congratulations on taking proactive measures to manage your blood pressure.\n\nYes, there are several ways to lower blood pressure naturally. Here are some tips that may help:\n\n1. **Maintain a healthy weight**: Excess weight can put extra strain on your heart and blood vessels, leading to higher blood pressure. Aim for a healthy body mass index (BMI) through a balanced diet and regular exercise.\n2. **Exercise regularly**: Regular physical activity can help lower blood pressure and improve overall cardiovascular health. Aim for at least 150 minutes of moderate-intensity exercise or 75 minutes of vigorous-intensity exercise per week.\n3. **Reduce sodium intake**: Consuming high amounts of sodium can increase blood pressure. Limit your daily sodium intake to less than 2,300 milligrams (mg) and ideally no more than 1,500 mg if you're at risk for high blood pressure.\n4. **Eat a balanced diet**: Focus on whole, unprocessed foods like fruits, vegetables, whole grains, lean proteins, and healthy fats. Avoid sugary drinks and foods high in saturated and trans fats.\n5. **Increase potassium intake**: Potassium can help lower blood pressure by balancing out the effects of sodium. Include potassium-rich foods like bananas, leafy greens, and sweet potatoes in your diet.\n6. **Stay hydrated**: Drinking enough water can help lower blood pressure by improving blood flow and reducing blood volume. Aim for at least eight glasses of water per day.\n7. **Manage stress**: Chronic stress can contribute to high blood pressure. Practice stress-reducing techniques like meditation, deep breathing, or yoga to help manage stress.\n8. **Get enough sleep**: Lack of sleep can increase blood pressure. Aim for 7-8 hours of sleep per night and prioritize a consistent sleep schedule.\n9. **Limit alcohol consumption**: Drinking too much alcohol can increase blood pressure. Limit your daily alcohol intake to no more than one drink for women and two drinks for men.\n10. **Quit smoking**: Smoking is a significant risk factor for high blood pressure. Quitting smoking can help lower blood pressure and improve overall health.\n\nRemember to consult with your healthcare provider before making any significant changes to your lifestyle or diet. They can provide personalized guidance and recommendations based on your individual health needs.\n\nIt's great that you're taking proactive steps to monitor and manage your blood pressure. Keep up the good work, and don't hesitate to reach out if you have any more questions or concerns!"}, 
        {"role": "user", "content": "I'm also experimenting with essential oils for stress relief, and I recently got a new diffuser on March 22nd. It's been a game-changer for unwinding before bed. I've been using a lavender and chamomile blend that I got from a local health food store."}, 
        {"role": "assistant", "content": "Essential oils can be a wonderful addition to your self-care routine! It's great that you're exploring natural ways to manage stress and promote relaxation.\n\nLavender and chamomile are both excellent choices for promoting calmness and reducing anxiety. The combination of the two can be particularly potent in promoting relaxation and improving sleep quality.\n\nUsing a diffuser is a great way to enjoy the benefits of essential oils, and it's wonderful that you've found it to be a game-changer for unwinding before bed. The gentle mist of the diffuser can help create a calming atmosphere, making it easier to relax and prepare for a restful night's sleep.\n\nRemember to always use high-quality, pure essential oils and follow the recommended usage guidelines to ensure safe and effective use. It's also a good idea to consult with a healthcare professional or a certified aromatherapist if you have any underlying health conditions or concerns.\n\nIn addition to lavender and chamomile, you might also consider exploring other essential oils that can promote relaxation and stress relief, such as:\n\n* Bergamot: Known for its calming and uplifting properties, bergamot can help reduce anxiety and stress.\n* Ylang-ylang: This sweet and soothing oil can help calm the mind and body, promoting feelings of relaxation and tranquility.\n* Frankincense: This earthy and woody oil can help reduce anxiety and stress, promoting a sense of calm and relaxation.\n\nRemember to always dilute essential oils in a carrier oil and perform a patch test before using them on your skin.\n\nIt's wonderful that you're taking proactive steps to prioritize your mental and emotional well-being. Keep exploring and finding what works best for you!"}, 
        {"role": "user", "content": "I've been meaning to get a flu shot, but I haven't gotten around to it yet. I need to schedule an appointment with my doctor for that. Do you have any tips on how to prepare for a flu shot appointment?"}, 
        {"role": "assistant", "content": "Getting a flu shot is an important step in protecting yourself and others from the flu. It's great that you're taking proactive steps to prioritize your health.\n\nPreparing for a flu shot appointment is relatively straightforward. Here are some tips to help you prepare:\n\n1. **Schedule your appointment**: As you mentioned, schedule an appointment with your doctor or a healthcare provider. You can also check with your local health department or pharmacy to see if they offer flu shots.\n2. **Check your insurance**: Verify that your insurance covers flu shots and what the copayment or coinsurance might be.\n3. **Dress appropriately**: Wear loose, easy-to-roll-up sleeves to make it easier to administer the shot.\n4. **Bring your insurance card and ID**: Make sure to bring your insurance card and a valid form of identification to your appointment.\n5. **Arrive early**: Plan to arrive 10-15 minutes before your scheduled appointment time to complete any necessary paperwork.\n6. **Be prepared for questions**: Your healthcare provider may ask you questions about your medical history, allergies, and any medications you're taking. Be prepared to provide this information.\n7. **Ask questions**: If you have any concerns or questions about the flu shot, be sure to ask your healthcare provider.\n8. **Plan for some downtime**: While the flu shot is a quick procedure, you may experience some minor side effects like soreness or redness at the injection site. Plan to take it easy for a few hours after the shot.\n\nSome general tips to keep in mind:\n\n* Flu shots are usually available from September to March, so it's best to get vaccinated early in the season.\n* The flu shot is recommended for everyone 6 months of age and older.\n* The flu shot cannot give you the flu, but it can take about two weeks for the vaccine to become effective.\n\nRemember, getting a flu shot is an important step in protecting yourself and others from the flu. By being prepared and taking proactive steps, you can help ensure a smooth and successful appointment.\n\nI hope this helps! If you have any more questions or concerns, feel free to ask."},
        {"role": "user", "content": "Can you give me numbered topics on radiation therapy that could be used to lecture a class on that subject ?"}, 
        {"role": "assistant", "content": "Sure, here are ten numbered topics on radiation therapy that could be used to lecture a class on that subject:\n\n1. Introduction to radiation therapy: history, principles, and applications\n2. Types of radiation therapy: external beam radiation therapy, internal radiation therapy, and systemic radiation therapy\n3. Radiation sources: photons, electrons, protons, and heavy ions\n4. Radiation delivery systems: linear accelerators, cyclotrons, brachytherapy sources, and radionuclide generators\n5. Radiation treatment planning: simulation, imaging, dose calculation, and quality assurance\n6. Radiation dose fractionation: conventional fractionation, hypofractionation, and stereotactic body radiation therapy\n7. Radiation side effects: acute and chronic effects, normal tissue toxicity, and late effects\n8. Radiation oncology team: radiation oncologists, medical physicists, radiation therapists, dosimetrists, and nurses\n9. Radiation safety: radiation protection, quality control, and emergency procedures\n10. Future directions in radiation therapy: advanced technologies, personalized medicine, and clinical trials"}, 
        {"role": "user", "content": "Pretend you are a teacher lecturing a class about topic 4, plan a class around that."}, 
        {"role": "assistant", "content": "Sure, here's a sample lesson plan for teaching Topic 4: Radiation Delivery Systems in Radiation Therapy:\n\nTitle: Radiation Delivery Systems in Radiation Therapy\n\nObjectives:\n\n* To introduce the different types of radiation delivery systems used in radiation therapy\n* To discuss the advantages and disadvantages of each type of delivery system\n* To provide an overview of the physics behind radiation delivery and the principles of dosimetry\n\nMaterials:\n\n* PowerPoint presentation\n* Handouts summarizing the main points of the lecture\n* Whiteboard and markers\n\nProcedure:\n\nI. Introduction (5 minutes)\n\n* Start the class by introducing the topic of radiation delivery systems and their importance in radiation therapy.\n* Ask the students to share any prior knowledge they may have about the different types of radiation delivery systems.\n\nII. Types of radiation delivery systems (20 minutes)\n\n* Use the PowerPoint presentation to introduce the different types of radiation delivery systems used in radiation therapy, including linear accelerators, cyclotrons, brachytherapy sources, and radionuclide generators.\n* For each type of delivery system, discuss the advantages and disadvantages, such as dose rate, beam energy, and treatment time.\n\nIII. Physics and dosimetry (20 minutes)\n\n* Explain the physics behind radiation delivery and the principles of dosimetry, including dose calculations, treatment planning, and quality assurance.\n* Use the whiteboard to illustrate key concepts and equations, such as the inverse square law and the depth-dose curve.\n\nIV. Case studies (10 minutes)\n\n* Use case studies to illustrate how different types of radiation delivery systems are used in clinical practice.\n* Ask the students to discuss how the choice of delivery system might impact the treatment plan and outcomes for each case.\n\nV. Conclusion (5 minutes)\n\n* Summarize the main points of the lecture and provide the handouts for the students to review.\n* Encourage the students to ask any remaining questions they may have about radiation delivery systems in radiation therapy.\n\nAssessment:\n\n* Give the students a short quiz or homework assignment to assess their understanding of the different types of radiation delivery systems and their advantages and disadvantages.\n* Ask the students to write a short reflection paper on how the choice of radiation delivery system might impact the success of a radiation therapy treatment plan."}, 
        {"role": "user", "content": "Can you write the texts for the PowerPoint presentation ?"}, 
        {"role": "assistant", "content": "Certainly! Here's an example PowerPoint presentation for Topic 4: Radiation Delivery Systems in Radiation Therapy:\n\nSlide 1: Radiation Delivery Systems in Radiation Therapy\n\nSlide 2: Introduction\n\n* Radiation delivery systems are the tools used to deliver radiation to the tumor site during radiation therapy.\n* The choice of delivery system depends on the tumor type, location, and stage, as well as the patient's overall health.\n\nSlide 3: Linear Accelerators (LINACs)\n\n* LINACs are the most common type of radiation delivery system used in radiation therapy.\n* They use electricity to produce high-energy X-rays or electrons that are directed at the tumor site.\n* Advantages: can be used to treat tumors in any part of the body, high dose rates, precise dose delivery.\n* Disadvantages: limited energy range, large equipment size, high cost.\n\nSlide 4: Cyclotrons\n\n* Cyclotrons are used to produce protons and other charged particles for radiation therapy.\n* They accelerate the particles to high energies and then direct them at the tumor site.\n* Advantages: high dose conformity, fewer side effects to healthy tissue, effective for some types of tumors.\n* Disadvantages: limited availability, high cost, large equipment size.\n\nSlide 5: Brachytherapy Sources\n\n* Brachytherapy involves placing a radioactive source directly into or near the tumor site.\n* Sources can be temporary or permanent and are usually inserted through a catheter or applicator.\n* Advantages: high dose rates, precise dose delivery, lower risk of side effects to healthy tissue.\n* Disadvantages: limited to certain types of tumors, risk of radiation exposure to healthcare workers, difficult to remove.\n\nSlide 6: Radionuclide Generators\n\n* Radionuclide generators are used to produce short-lived isotopes for imaging and therapy.\n* They consist of a parent isotope that decays into a daughter isotope, which can be used for therapy.\n* Advantages: relatively low cost, high purity of isotopes, convenient to use.\n* Disadvantages: limited availability of isotopes, risk of radiation exposure to healthcare workers, limited to certain types of tumors.\n\nSlide 7: Physics and Dosimetry\n\n* The physics of radiation delivery involves understanding how the radiation beam interacts with the patient's tissue.\n* Dosimetry is the process of calculating the radiation dose delivered to the tumor and surrounding tissue.\n* Key factors in dosimetry include beam energy, dose rate, and beam shape.\n\nSlide 8: Case Studies\n\n* Case studies can help illustrate how different types of radiation delivery systems are used in clinical practice.\n* For example, a LINAC might be used to treat a tumor in the brain, while a brachytherapy source might be used to treat a tumor in the cervix.\n\nSlide 9: Conclusion\n\n* Radiation delivery systems play a crucial role in radiation therapy.\n* The choice of delivery system depends on the tumor type, location, and stage, as well as the patient's overall health.\n* By understanding the advantages and disadvantages of each type of delivery system, radiation therapists can make informed treatment decisions."}, 
        {"role": "user", "content": "Can you plan a class about the secondary effects of external radiation therapy ?"}, 
        {"role": "assistant", "content": "Sure, here's a sample lesson plan for teaching about the secondary effects of external radiation therapy:\n\nTitle: Secondary Effects of External Radiation Therapy\n\nObjectives:\n\n* To introduce the concept of secondary effects of external radiation therapy\n* To discuss the different types of secondary effects and their causes\n* To provide an overview of the strategies for managing secondary effects\n\nMaterials:\n\n* PowerPoint presentation\n* Handouts summarizing the main points of the lecture\n* Whiteboard and markers\n\nProcedure:\n\nI. Introduction (5 minutes)\n\n* Start the class by introducing the topic of secondary effects of external radiation therapy and their importance in patient care.\n* Ask the students to share any prior knowledge they may have about the secondary effects of radiation therapy.\n\nII. Types of secondary effects (20 minutes)\n\n* Use the PowerPoint presentation to introduce the different types of secondary effects of external radiation therapy, including acute and chronic effects.\n* For each type of secondary effect, discuss the common symptoms, causes, and timelines.\n\nIII. Causes of secondary effects (20 minutes)\n\n* Discuss the underlying causes of secondary effects of external radiation therapy, such as damage to normal tissue and organs.\n* Use the whiteboard to illustrate key concepts, such as radiation-induced DNA damage and the role of inflammation in secondary effects.\n\nIV. Strategies for managing secondary effects (20 minutes)\n\n* Provide an overview of the strategies for managing secondary effects of external radiation therapy, including supportive care, medications, and radiation therapy techniques.\n* Discuss the importance of multidisciplinary care in managing secondary effects, including the roles of radiation oncologists, nurses, and other healthcare professionals.\n\nV. Case studies (10 minutes)\n\n* Use case studies to illustrate how different strategies for managing secondary effects are used in clinical practice.\n* Ask the students to discuss how the choice of management strategy might impact the patient's quality of life and treatment outcomes for each case.\n\nVI. Conclusion (5 minutes)\n\n* Summarize the main points of the lecture and provide the handouts for the students to review.\n* Encourage the students to ask any remaining questions they may have about secondary effects of external radiation therapy.\n\nAssessment:\n\n* Give the students a short quiz or homework assignment to assess their understanding of the different types of secondary effects of external radiation therapy and their causes.\n* Ask the students to write a short reflection paper on the importance of managing secondary effects in patient care."}]
    ]
}


for turn_messages in session["turns"]:
    timestamp = session["timestamp"]
    for msg in turn_messages:
        msg["time_stamp"] = timestamp
        
    store_result = lightmem.add_memory(
        messages=turn_messages,
        force_segment=True,
        force_extract=True
    )
```

### Offline Update
```python
lightmem.construct_update_queue_all_entries()
lightmem.offline_update_all_entries(score_threshold=0.8)
``` 

### Retrieve Memory
```python
question = "What is the name of my dog?"
related_memories = lightmem.retrieve(question, limit=5)
print(related_memories)
``` 

<span id='configuration'/>

## ⚙️ Configuration

All behaviors of LightMem are controlled via the BaseMemoryConfigs configuration class. Users can customize aspects like pre-processing, memory extraction, retrieval strategy, and update mechanisms by providing a custom configuration.
#### Key Configuration Options (Usage)

| Option                    | Default                                     | Usage (allowed values and behavior) |
| :---                      | :---                                        | :--- |
| `pre_compress`        | `False`                                     | True / False. If True, input messages are pre-compressed using the `pre_compressor` configuration before being stored. This reduces storage and indexing cost but may remove fine-grained details. If False, messages are stored without pre-compression. |
| `pre_compressor`      | `None`                                      | dict / object. Configuration for the pre-compression component (`PreCompressorConfig`) with fields like `model_name` (e.g., `llmlingua-2`, `entropy_compress`) and `configs` (model-specific parameters). Effective only when `pre_compress=True`. |
| `topic_segment`       | `False`                                     | True / False. Enables topic-based segmentation of long conversations. When True, long conversations are split into topic segments and each segment can be indexed/stored independently (requires `topic_segmenter`). When False, messages are stored sequentially. |
| `precomp_topic_shared`| `False`                                     | True / False. If True, pre-compression and topic segmentation can share intermediate results to avoid redundant processing. May improve performance but requires careful configuration to avoid cross-topic leakage. |
| `topic_segmenter`     | `None`                                      | dict / object. Configuration for topic segmentation (`TopicSegmenterConfig`), including `model_name` and `configs` (segment length, overlap, etc.). Used when `topic_segment=True`. |
| `messages_use`        | `'user_only'`                               | `'user_only'` / `'assistant_only'` / `'hybrid'`. Controls which messages are used to generate metadata and summaries: `user_only` uses user inputs, `assistant_only` uses assistant responses, `hybrid` uses both. Choosing `hybrid` increases processing but yields richer context. |
| `metadata_generate`   | `True`                                      | True / False. If True, metadata such as keywords and entities are extracted and stored to support attribute-based and filtered retrieval. If False, no metadata extraction occurs. |
| `text_summary`        | `True`                                      | True / False. If True, a text summary is generated and stored alongside the original text (reduces retrieval cost and speeds review). If False, only the original text is stored. Summary quality depends on `memory_manager`. |
| `memory_manager`      | `MemoryManagerConfig()`                     | dict / object. Controls the model used to generate summaries and metadata (`MemoryManagerConfig`), e.g., `model_name` (`openai`, `deepseek`) and `configs`. Changing this affects summary style, length, and cost. |
| `extract_threshold`   | `0.5`                                       | float (0.0 - 1.0). Threshold used to decide whether content is important enough to be extracted as metadata or highlight. Higher values (e.g., 0.8) mean more conservative extraction; lower values (e.g., 0.2) extract more items (may increase noise). |
| `index_strategy`      | `None`                                      | `'embedding'` / `'context'` / `'hybrid'` / `None`. Determines how memories are indexed: 'embedding' uses vector-based indexing (requires embedders/retriever) for semantic search; 'context' uses text-based/contextual retrieval (requires context_retriever) for keyword/document similarity; and 'hybrid' combines context filtering and vector reranking for robustness and higher accuracy.
| `text_embedder`       | `None`                                      | dict / object. Configuration for text embedding model (`TextEmbedderConfig`) with `model_name` (e.g., `huggingface`) and `configs` (batch size, device, embedding dim). Required when `index_strategy` or `retrieve_strategy` includes `'embedding'`. |
| `multimodal_embedder` | `None`                                      | dict / object. Configuration for multimodal/image embedder (`MMEmbedderConfig`). Used for non-text modalities. |
| `history_db_path`     | `os.path.join(lightmem_dir, "history.db")`  | str. Path to persist conversation history and lightweight state. Useful to restore state across restarts. |
| `retrieve_strategy`   | `'embedding'`                               | `'embedding'` / `'context'` / `'hybrid'`. Strategy used at query time to fetch relevant memories. Pick based on data and query type: semantic queries -> `'embedding'`; keyword/structured queries -> `'context'`; mixed -> `'hybrid'`. |
| `context_retriever`   | `None`                                      | dict / object. Configuration for context-based retriever (`ContextRetrieverConfig`), e.g., `model_name='BM25'` and `configs` like `top_k`. Used when `retrieve_strategy` includes `'context'`. |
| `embedding_retriever` | `None`                                      | dict / object. Vector store configuration (`EmbeddingRetrieverConfig`), e.g., `model_name='qdrant'` and connection/index params. Used when `retrieve_strategy` includes `'embedding'`. |
| `update`              | `'offline'`                                 | `'online'` / `'offline'`. `'online'`: update memories immediately after each interaction (low latency for fresh memories but higher operational cost). `'offline'`: batch or scheduled updates to save cost and aggregate changes. |
| `kv_cache`            | `False`                                     | True / False. If True, attempt to precompute and persist model KV caches to accelerate repeated LLM calls (requires support from the LLM runtime and may increase storage). Uses `kv_cache_path` to store cache. |
| `kv_cache_path`       | `os.path.join(lightmem_dir, "kv_cache.db")` | str. File path for KV cache storage when `kv_cache=True`. |
| `graph_mem`           | `False`                                     | True / False. When True, some memories will be organized as a graph (nodes and relationships) to support complex relation queries and reasoning. Requires additional graph processing/storage. |
| `version`             | `'v1.1'`                                    | str. Configuration/API version. Only change if you know compatibility implications. |


## 👥 Contributors
We welcome contributions from the community! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
