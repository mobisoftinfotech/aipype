"""pytest configuration for integration tests - NO AUTO-MOCKING."""

import os
import pytest
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configurable model for integration tests
INTEGRATION_TEST_MODEL = os.getenv("INTEGRATION_TEST_MODEL", "gemma3:1b")


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Check if Ollama service is available."""
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def integration_test_model() -> str:
    """Get the model to use for integration tests."""
    return INTEGRATION_TEST_MODEL


@pytest.fixture(scope="session")
def test_model_available(ollama_available: bool, integration_test_model: str) -> bool:
    """Check if the configured integration test model is available in Ollama."""
    if not ollama_available:
        return False

    try:
        # Check if model is available
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(
                integration_test_model in model.get("name", "") for model in models
            )
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="session")
def gemma_model_available(ollama_available: bool) -> bool:
    """Check if gemma:7b model is available in Ollama.

    DEPRECATED: Use test_model_available instead for configurable model checking.
    """
    if not ollama_available:
        return False

    try:
        # Check if model is available
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any("gemma:7b" in model.get("name", "") for model in models)
        return False
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture
def ollama_config(integration_test_model: str) -> Dict[str, Any]:
    """Ollama configuration for integration tests with configurable model."""
    return {
        "llm_provider": "ollama",
        "llm_model": integration_test_model,
        "api_base": "http://localhost:11434",
        "temperature": 0.3,
        "max_tokens": 500,
    }


@pytest.fixture
def realistic_articles_short() -> List[Dict[str, Any]]:
    """Provide shorter realistic articles for basic testing."""
    return [
        {
            "title": "The Rise of Edge Computing in IoT",
            "url": "https://example.com/edge-computing",
            "content": """Edge computing represents a paradigm shift in how we process and analyze data in the Internet of Things (IoT) ecosystem. Unlike traditional cloud computing, where data is sent to centralized data centers for processing, edge computing brings computation and data storage closer to the location where it is needed.

This approach offers several significant advantages. First, it dramatically reduces latency by processing data locally rather than sending it across networks to distant servers. This is crucial for time-sensitive applications like autonomous vehicles, industrial automation, and real-time monitoring systems.

Second, edge computing enhances privacy and security by keeping sensitive data local. Instead of transmitting raw data across networks where it could be intercepted, only processed results or aggregated information needs to be shared.

The technology also improves reliability and resilience. When processing happens at the edge, systems can continue to function even if connectivity to the central cloud is interrupted. This is particularly important for critical infrastructure and mission-critical applications.

From a bandwidth perspective, edge computing is a game-changer. By processing data locally and only sending relevant insights to the cloud, it significantly reduces the amount of data that needs to be transmitted. This not only saves bandwidth costs but also reduces network congestion.

Industries are rapidly adopting edge computing solutions. Manufacturing facilities use edge devices to monitor equipment health and predict maintenance needs. Smart cities deploy edge computing for traffic optimization and energy management. Healthcare organizations leverage edge computing for real-time patient monitoring and diagnostic systems.

The future of edge computing looks promising, with 5G networks enabling even more sophisticated edge applications. As IoT devices become more prevalent and AI capabilities become more accessible, edge computing will play an increasingly vital role in our connected world.""",
        },
        {
            "title": "Sustainable Software Development Practices",
            "url": "https://example.com/sustainable-dev",
            "content": """The software development industry is increasingly recognizing its environmental impact and the need for sustainable practices. As digital transformation accelerates across industries, the carbon footprint of software development and deployment has become a significant concern.

Green coding practices focus on writing efficient code that minimizes computational resources and energy consumption. This includes optimizing algorithms, reducing memory usage, and eliminating unnecessary operations. Developers are learning to measure the energy efficiency of their code and make informed decisions about trade-offs between features and environmental impact.

Cloud optimization plays a crucial role in sustainable software development. By choosing appropriate instance sizes, implementing auto-scaling, and using serverless architectures where appropriate, organizations can significantly reduce their energy consumption. The selection of data centers powered by renewable energy sources further amplifies the environmental benefits.

The concept of digital minimalism in software design is gaining traction. This approach emphasizes building only the features that are truly necessary, reducing both development effort and the resulting software's resource requirements. Clean, minimal interfaces not only improve user experience but also require less processing power to render and maintain.

Sustainable development lifecycle practices include considering the environmental impact during the planning phase, implementing continuous monitoring of resource usage, and regularly auditing applications for optimization opportunities. Teams are adopting metrics that track energy consumption alongside traditional performance indicators.

Open source collaboration contributes to sustainability by reducing duplicated effort across the industry. When developers share solutions and build upon existing work, it prevents the creation of redundant software and promotes more efficient resource utilization across the entire ecosystem.

The rise of edge computing also supports sustainability goals by reducing data transmission requirements and enabling more efficient distributed processing. By processing data closer to its source, applications can reduce bandwidth usage and improve overall system efficiency.

Looking ahead, artificial intelligence and machine learning tools are being developed to automatically optimize code for energy efficiency, suggesting improvements and identifying resource-intensive operations that could be streamlined.""",
        },
    ]


@pytest.fixture
def realistic_articles_long() -> List[Dict[str, Any]]:
    """Provide long-form realistic articles (2000-5000 words) for comprehensive testing."""
    return [
        {
            "title": "The Evolution and Future of Artificial Intelligence in Healthcare",
            "url": "https://example.com/ai-healthcare-future",
            "content": """Artificial intelligence has emerged as one of the most transformative technologies in healthcare, revolutionizing how we diagnose, treat, and manage diseases. From its humble beginnings in expert systems to today's sophisticated machine learning algorithms, AI has evolved to become an indispensable tool in modern medicine.

The journey of AI in healthcare began in the 1970s with expert systems like MYCIN, which was designed to identify bacteria causing severe infections and recommend antibiotics. These early systems relied on rule-based logic and knowledge bases created by human experts. While limited in scope, they demonstrated the potential for computers to assist in medical decision-making.

The 1990s and early 2000s saw the development of more sophisticated algorithms and the introduction of machine learning techniques. However, it wasn't until the explosion of big data and computing power in the 2010s that AI truly began to show its potential in healthcare applications.

Today, AI applications in healthcare span numerous domains, each addressing critical challenges in patient care and medical practice. Medical imaging has been one of the most successful areas of AI implementation. Deep learning algorithms can now analyze radiological images with accuracy that matches or exceeds that of human radiologists in many cases. For instance, AI systems can detect diabetic retinopathy in retinal photographs, identify skin cancer in dermatological images, and spot early signs of lung cancer in CT scans.

These imaging applications have proven particularly valuable in areas with shortages of specialist physicians. In developing countries or rural areas where access to expert radiologists is limited, AI-powered diagnostic tools can provide initial screenings and flag cases that require immediate attention. This democratization of expert-level diagnostic capabilities has the potential to save countless lives by enabling early detection of serious conditions.

Drug discovery and development represent another frontier where AI is making significant contributions. The traditional drug development process is notoriously slow and expensive, often taking 10-15 years and costing billions of dollars to bring a new drug to market. AI is accelerating this process by identifying potential drug compounds, predicting their effectiveness, and optimizing clinical trial design.

Machine learning algorithms can analyze vast databases of molecular structures, genetic information, and clinical data to identify promising drug targets and predict how different compounds might interact with these targets. This computational approach can significantly reduce the number of compounds that need to be synthesized and tested in the laboratory, saving both time and resources.

Personalized medicine is perhaps one of the most exciting applications of AI in healthcare. By analyzing a patient's genetic profile, medical history, lifestyle factors, and real-time health data, AI systems can recommend personalized treatment plans that are tailored to the individual's unique characteristics. This approach moves away from the traditional one-size-fits-all model of medicine toward more precise and effective treatments.

For example, in oncology, AI algorithms can analyze tumor genetics and patient characteristics to predict which treatments are most likely to be effective for a particular patient. This personalized approach not only improves treatment outcomes but also reduces the risk of adverse effects by avoiding treatments that are unlikely to work for a specific individual.

Electronic health records (EHRs) and clinical decision support systems have been enhanced significantly through AI integration. Natural language processing algorithms can extract valuable information from unstructured clinical notes, enabling more comprehensive analysis of patient data. AI-powered clinical decision support systems can alert healthcare providers to potential drug interactions, suggest differential diagnoses, and recommend evidence-based treatment protocols.

Predictive analytics powered by AI is helping healthcare organizations identify patients at risk of developing complications, hospital readmissions, or adverse events. By analyzing patterns in patient data, these systems can flag high-risk individuals for early intervention, potentially preventing serious complications and reducing healthcare costs.

The integration of AI with wearable devices and remote monitoring technologies is enabling continuous health monitoring and early disease detection. Smart watches and fitness trackers equipped with AI algorithms can detect irregular heart rhythms, monitor blood glucose levels, and track various health metrics. This continuous monitoring capability allows for the early detection of health issues and enables proactive healthcare management.

Telemedicine has been another area significantly enhanced by AI technologies. AI-powered chatbots and virtual assistants can conduct initial patient assessments, triage cases based on severity, and provide basic health information. During the COVID-19 pandemic, these tools proved invaluable in managing the increased demand for healthcare services while minimizing the risk of viral transmission.

Despite these remarkable advances, the implementation of AI in healthcare faces several significant challenges. Data privacy and security concerns are paramount, as healthcare AI systems require access to sensitive patient information. Ensuring compliance with regulations like HIPAA while enabling AI development and deployment requires careful balance and robust security measures.

The issue of algorithmic bias represents another critical challenge. AI systems are only as good as the data they are trained on, and if training data is not representative of diverse patient populations, the resulting algorithms may perform poorly for underrepresented groups. This can exacerbate existing healthcare disparities and create new forms of inequality in medical care.

Regulatory approval and validation of AI systems in healthcare present unique challenges. Traditional clinical trial methodologies may not be well-suited for evaluating AI algorithms, which can be continuously updated and improved. Regulatory bodies like the FDA are developing new frameworks for approving AI-based medical devices, but the process is still evolving.

Integration with existing healthcare systems and workflows remains a significant hurdle. Many healthcare organizations operate with legacy systems that may not be compatible with modern AI technologies. Successfully implementing AI solutions requires not only technical integration but also changes to established workflows and practices.

The training and education of healthcare professionals in AI technologies is crucial for successful implementation. Healthcare providers need to understand the capabilities and limitations of AI systems to use them effectively and safely. This requires ongoing education and training programs to keep pace with rapidly evolving AI technologies.

Looking toward the future, several emerging trends and technologies promise to further transform healthcare through AI. Quantum computing may eventually enable more complex simulations and optimizations that are currently computationally infeasible. This could accelerate drug discovery, enable more sophisticated predictive models, and unlock new applications in precision medicine.

The integration of AI with genomics and proteomics is likely to yield deeper insights into disease mechanisms and treatment targets. As our understanding of the genetic basis of diseases improves and genomic sequencing becomes more affordable, AI will play a crucial role in interpreting this complex biological data and translating it into clinical applications.

Robotic surgery enhanced by AI is becoming more precise and less invasive. AI algorithms can provide real-time guidance during surgical procedures, helping surgeons navigate complex anatomy and avoid critical structures. As these technologies mature, they may enable new types of minimally invasive procedures and improve surgical outcomes.

The concept of digital twins in healthcare, where AI models simulate individual patients' physiology and disease progression, holds promise for personalized medicine and treatment optimization. These digital representations could be used to test different treatment strategies virtually before implementing them in real patients.

Federated learning approaches are being developed to enable AI model training across multiple healthcare institutions without sharing sensitive patient data. This approach could accelerate AI development while preserving patient privacy and enabling more diverse and representative training datasets.

The future success of AI in healthcare will depend on addressing current challenges while continuing to innovate and develop new applications. This will require collaboration between technologists, healthcare providers, regulators, and patients to ensure that AI technologies are developed and deployed in ways that are safe, effective, equitable, and aligned with healthcare goals.

As AI continues to evolve, it has the potential to make healthcare more accessible, affordable, and effective. However, realizing this potential will require careful consideration of ethical implications, ongoing investment in research and development, and a commitment to ensuring that the benefits of AI are distributed equitably across all populations.

The transformation of healthcare through AI is not just a technological revolution but a fundamental shift in how we approach human health and well-being. By harnessing the power of artificial intelligence responsibly and thoughtfully, we can create a future where healthcare is more precise, personalized, and accessible to all.""",
        },
        {
            "title": "Blockchain Technology Beyond Cryptocurrency: Enterprise Applications and Future Potential",
            "url": "https://example.com/blockchain-enterprise",
            "content": """Blockchain technology, originally conceived as the underlying infrastructure for Bitcoin, has evolved far beyond its cryptocurrency origins to become a transformative force across numerous industries. While digital currencies remain the most visible application of blockchain technology, its potential for revolutionizing enterprise operations, supply chain management, healthcare, finance, and governance is increasingly being recognized and implemented across various sectors.

At its core, blockchain represents a paradigm shift from centralized to decentralized systems of record-keeping and trust establishment. The technology creates an immutable, distributed ledger that maintains a continuously growing list of records, called blocks, which are linked and secured using cryptography. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data, creating a chain that is extremely difficult to alter retroactively without changing all subsequent blocks.

The fundamental characteristics that make blockchain revolutionary include its decentralized nature, which eliminates the need for traditional intermediaries; its transparency, which allows all participants to view the same information; its immutability, which prevents unauthorized changes to historical records; and its security, which is maintained through cryptographic techniques and consensus mechanisms.

Supply chain management has emerged as one of the most promising and widely implemented applications of blockchain technology beyond cryptocurrency. Traditional supply chains often lack transparency and traceability, making it difficult to verify the authenticity and origin of products. This opacity can lead to issues such as counterfeiting, fraud, and inability to quickly identify and address problems when they arise.

Blockchain-based supply chain solutions create an end-to-end transparent and traceable system where every step of a product's journey is recorded on an immutable ledger. From raw material sourcing to manufacturing, distribution, and retail, each transaction and transfer of ownership is documented with timestamps and digital signatures. This level of transparency enables consumers to verify the authenticity and ethical sourcing of products, while businesses can quickly identify and address issues such as contamination or defects.

Walmart has been a pioneer in this space, implementing blockchain technology to track food products from farm to store shelves. Their system can trace the origin of contaminated food within seconds rather than days or weeks, significantly reducing the scope and impact of food safety incidents. Similarly, De Beers uses blockchain to track diamonds from mining to retail, ensuring they are conflict-free and authentic.

Healthcare represents another sector where blockchain technology is demonstrating significant potential. Patient data management, clinical trial integrity, drug traceability, and insurance claim processing are areas where blockchain can address existing challenges and improve outcomes.

In patient data management, blockchain can create secure, interoperable health records that patients control and can share selectively with healthcare providers. This addresses current issues with fragmented medical records and gives patients greater control over their health information. The immutable nature of blockchain ensures the integrity of medical records while smart contracts can automate consent management and access controls.

Clinical trial management benefits from blockchain's ability to ensure data integrity and prevent tampering with trial results. All trial data can be recorded on the blockchain with timestamps, making it impossible to alter results after the fact. This transparency and immutability can increase trust in clinical trial outcomes and accelerate the approval process for new treatments.

Drug traceability is another critical application, as pharmaceutical counterfeiting is a major global problem that threatens patient safety. Blockchain-based systems can track medications from manufacturing through distribution to dispensing, creating an unbreakable chain of custody that makes counterfeiting extremely difficult.

Financial services beyond cryptocurrency are being transformed by blockchain technology through applications such as trade finance, cross-border payments, identity verification, and smart contracts. Traditional trade finance involves numerous intermediaries, complex documentation, and lengthy processing times. Blockchain can streamline these processes by creating a single, shared ledger that all parties can access, reducing the need for intermediaries and accelerating transaction processing.

Cross-border payments, which traditionally take days and involve multiple intermediaries charging fees, can be executed more efficiently using blockchain technology. While not as fast as some centralized systems, blockchain-based cross-border payments can reduce costs and increase transparency compared to traditional correspondent banking networks.

Identity verification and management represent another significant application area. Traditional identity systems are often fragmented and vulnerable to data breaches. Blockchain-based identity solutions can give individuals greater control over their personal information while providing secure and verifiable digital identities. Self-sovereign identity systems built on blockchain allow individuals to manage their own identity credentials without relying on centralized authorities.

Smart contracts, which are self-executing contracts with terms directly written into code, automate many financial and business processes. These contracts automatically execute when predetermined conditions are met, reducing the need for intermediaries and minimizing the potential for disputes. Smart contracts are being used for insurance claims processing, loan origination, and various other financial services.

Real estate transactions are being revolutionized by blockchain technology through applications such as property title management, fractional ownership, and transparent property histories. Property title fraud is a significant problem in many jurisdictions, and blockchain-based title management systems can create immutable records of property ownership that are extremely difficult to forge or manipulate.

Fractional ownership platforms built on blockchain technology allow multiple investors to own shares of real estate properties, making real estate investment more accessible and liquid. These platforms use tokens to represent ownership shares and can automate dividend distributions and ownership transfers through smart contracts.

The energy sector is exploring blockchain applications for peer-to-peer energy trading, renewable energy certificate management, and grid management. Peer-to-peer energy trading platforms allow individuals with solar panels or other renewable energy sources to sell excess energy directly to neighbors, bypassing traditional utility companies. Blockchain technology can facilitate these transactions securely and transparently while automating payment processing.

Renewable energy certificates, which verify that energy was generated from renewable sources, can be managed more efficiently using blockchain technology. This creates a more transparent and trustworthy system for tracking and trading renewable energy credits.

Voting and governance systems are being enhanced through blockchain technology to increase transparency, security, and accessibility. Traditional voting systems often face challenges related to security, transparency, and accessibility. Blockchain-based voting systems can provide verifiable and immutable records of votes while maintaining voter privacy through cryptographic techniques.

Several countries and organizations have piloted blockchain-based voting systems, though widespread adoption faces regulatory and technical challenges. The technology shows promise for both political elections and corporate governance applications.

Despite these promising applications, blockchain technology faces several significant challenges that must be addressed for widespread enterprise adoption. Scalability remains a primary concern, as most blockchain networks can process far fewer transactions per second compared to traditional centralized systems. Bitcoin, for example, can process approximately 7 transactions per second, while Visa's network can handle over 24,000 transactions per second.

Energy consumption is another major challenge, particularly for blockchain networks that use proof-of-work consensus mechanisms. Bitcoin's energy consumption has been criticized for its environmental impact, though newer consensus mechanisms like proof-of-stake are more energy-efficient.

Regulatory uncertainty poses challenges for blockchain adoption, as many jurisdictions lack clear regulatory frameworks for blockchain applications. This uncertainty can make organizations hesitant to invest heavily in blockchain solutions, as future regulatory changes could affect their viability.

Integration with existing systems and processes can be complex and costly. Many organizations operate with legacy systems that may not be compatible with blockchain technology, requiring significant investment in infrastructure upgrades and process redesign.

User experience and technical complexity remain barriers to adoption, as blockchain applications often require technical knowledge that many users lack. Improving user interfaces and simplifying interactions with blockchain systems are important for broader adoption.

The future of blockchain technology in enterprise applications looks promising, with several trends and developments likely to drive further adoption and innovation. Interoperability solutions are being developed to allow different blockchain networks to communicate and share information, which could unlock new applications and increase the overall utility of blockchain technology.

Central bank digital currencies (CBDCs) are being explored by numerous countries and could significantly increase familiarity with and adoption of blockchain technology. These government-issued digital currencies could serve as a bridge between traditional financial systems and blockchain-based systems.

Integration with other emerging technologies such as artificial intelligence, Internet of Things devices, and 5G networks could create new applications and use cases for blockchain technology. For example, IoT devices could automatically record data on blockchain networks, while AI systems could analyze blockchain data to identify patterns and optimize processes.

The development of more sustainable consensus mechanisms and layer-2 scaling solutions is addressing some of the current limitations of blockchain technology. These technical improvements could make blockchain more viable for large-scale enterprise applications.

As blockchain technology matures and these challenges are addressed, we can expect to see increased adoption across various industries. The technology's ability to create trust, transparency, and efficiency in systems that previously relied on intermediaries or centralized authorities positions it as a fundamental technology for the digital economy.

The transformation brought by blockchain extends beyond individual applications to represent a shift toward more decentralized, transparent, and democratized systems. Organizations that understand and effectively implement blockchain technology will be well-positioned to take advantage of these changes and create competitive advantages in their respective markets.

Success in implementing blockchain technology requires a clear understanding of the specific problems it solves, careful consideration of technical and regulatory challenges, and a strategic approach to integration with existing systems and processes. As the technology continues to evolve and mature, its potential to transform enterprise operations and create new business models will only continue to grow.""",
        },
    ]


@pytest.fixture
def skip_if_ollama_unavailable(ollama_available: bool) -> None:
    """Skip test if Ollama is not available."""
    if not ollama_available:
        pytest.skip("Ollama service is not available")


@pytest.fixture
def skip_if_test_model_unavailable(
    test_model_available: bool, integration_test_model: str
) -> None:
    """Skip test if the configured integration test model is not available."""
    if not test_model_available:
        pytest.skip(f"{integration_test_model} model is not available in Ollama")


@pytest.fixture
def skip_if_gemma_unavailable(gemma_model_available: bool) -> None:
    """Skip test if gemma:7b model is not available.

    DEPRECATED: Use skip_if_test_model_unavailable instead for configurable model checking.
    """
    if not gemma_model_available:
        pytest.skip("gemma:7b model is not available in Ollama")


def pytest_configure(config: Any) -> None:
    """Configure pytest with integration test markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring external services",
    )
    config.addinivalue_line("markers", "ollama: mark test as requiring Ollama service")
    config.addinivalue_line("markers", "slow: mark test as slow running (>10 seconds)")
    config.addinivalue_line("markers", "gmail: mark test as requiring Gmail API access")


def pytest_collection_modifyitems(config: Any, items: List[Any]) -> None:
    """Modify test collection to add markers for integration tests."""
    for item in items:
        # Add integration marker to all tests in integration_tests directory
        if "integration_tests" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
