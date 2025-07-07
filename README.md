# Poker AI Assistant

Um assistente de poker inteligente que combina visão computacional para detectar informações da mesa de poker (PokerStars) com análise de IA para fornecer recomendações de ação.

## 🚀 Funcionalidades

- **Detecção Visual**: Captura e analisa screenshots da mesa de poker
- **Reconhecimento de Cartas**: Detecta cartas do jogador e cartas comunitárias
- **Análise de IA**: Usa LLM (GPT-4) para análise estratégica de situações de poker
- **Recomendações Estruturadas**: Fornece dados organizados para GUI
- **Interface Gráfica**: Exemplo de GUI usando tkinter

## 📋 Pré-requisitos

- Python 3.8+
- OpenAI API Key (para análise de IA)
- Tesseract OCR instalado no sistema

### Instalação do Tesseract

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Baixe e instale de: https://github.com/UB-Mannheim/tesseract/wiki

## 🛠️ Instalação

1. Clone o repositório:

```bash
git clone <repository-url>
cd PokerAI
```

2. Crie um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou
venv\Scripts\activate  # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure sua API key do OpenAI:

```bash
export OPENAI_API_KEY="sua-api-key-aqui"
```

## 🎯 Como Usar

### 1. Teste do Módulo de Visão

Teste a detecção de elementos da mesa:

```bash
python test_vision.py
```

### 2. Teste do AI Advisor

Teste a análise de IA (com ou sem API key):

```bash
# Com API key configurada
python test_ai_advisor.py

# Modo fallback (sem API key)
python test_ai_advisor.py --fallback
```

### 3. Sistema Completo

Analise uma situação completa:

```bash
python src/main_poker_assistant.py imagem_tela.png
```

### 4. Interface Gráfica

Execute a GUI de exemplo:

```bash
python examples/gui_example.py
```

## 📁 Estrutura do Projeto

```
PokerAI/
├── src/
│   ├── core/                 # Motor de poker
│   ├── vision/               # Detecção visual
│   ├── ai/                   # Análise de IA
│   └── main_poker_assistant.py  # Sistema principal
├── config/
│   └── table_regions.json    # Configuração de regiões
├── examples/
│   └── gui_example.py        # Exemplo de GUI
├── test_vision.py            # Teste de visão
├── test_ai_advisor.py        # Teste de IA
└── requirements.txt
```

## 🔧 Configuração

### Calibração de Regiões

O sistema precisa ser calibrado para detectar corretamente os elementos da mesa. Use `test_vision.py` para ajustar as coordenadas em `config/table_regions.json`.

### Formato de Dados

O sistema retorna dados estruturados para fácil integração com GUI:

```python
{
    "table_info": {
        "pot_size": 80.0,
        "current_bet": 10.0,
        "street": "flop"
    },
    "ai_analysis": {
        "primary_action": {
            "action": "call",
            "confidence": 0.85,
            "reasoning": "Strong drawing hand with good pot odds...",
            "risk_level": "medium",
            "expected_value": 15.5
        },
        "hand_info": {
            "strength": "strong",
            "pot_odds": 8.0,
            "stack_to_pot_ratio": 1.25
        },
        "alternatives": [...]
    },
    "cards": {
        "hole_cards": ["AS", "KH"],
        "community_cards": ["QD", "JC", "10H"]
    },
    "metrics": {
        "detection_confidence": 0.95,
        "ai_confidence": 0.85,
        "action_color": "#ffaa00",
        "confidence_color": "#44ff44"
    }
}
```

## 🎨 Exemplo de GUI

O arquivo `examples/gui_example.py` demonstra como criar uma interface gráfica que:

- Permite selecionar imagens da mesa
- Exibe recomendações da IA com cores
- Mostra métricas e informações da mesa
- Apresenta dados brutos para debugging

## 🔍 Detecção Visual

O sistema detecta:

- **Tamanho do pote**: Valor total no centro da mesa
- **Cartas do jogador**: Suas duas cartas
- **Cartas comunitárias**: Flop, turn, river
- **Apostas atuais**: Valor para dar call/raise

## 🤖 Análise de IA

O AI Advisor considera:

- Força da mão atual
- Pot odds e implied odds
- Posição na mesa
- Histórico de ações
- Tamanho das stacks
- Street atual (preflop, flop, turn, river)

## 🚨 Tratamento de Erros

O sistema inclui:

- Fallback quando API não está disponível
- Validação de dados detectados
- Logging detalhado para debugging
- Tratamento de erros de OCR

## 📝 Logs

O sistema gera logs em `analysis_log.json` com:

- Timestamp da análise
- Dados detectados
- Recomendações da IA
- Métricas calculadas

## 🔧 Desenvolvimento

### Pre-commit Hooks

O projeto usa pre-commit hooks para qualidade de código:

```bash
pre-commit install
```

### Testes

Execute os testes:

```bash
pytest
```

### Formatação

Formate o código:

```bash
black src/ tests/
isort src/ tests/
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## ⚠️ Disclaimer

Este software é para fins educacionais e de entretenimento. O uso em jogos reais de poker pode violar os termos de serviço das plataformas de poker. Use por sua conta e risco.
