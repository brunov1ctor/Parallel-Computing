# Túnel de Vento Virtual 3D Interativo com Lattice Boltzmann Method

**Disciplina:** INE410129 — Computação Paralela
**Autor:** Bruno Victor
**Etapa:** Assessment 1 (A1) — Proposta de Projeto

---

## 1. Motivação

Simulações de dinâmica dos fluidos computacional (*Computational Fluid Dynamics* — CFD) são uma das aplicações mais tradicionais e computacionalmente exigentes da computação de alto desempenho, usadas em engenharia aeroespacial, automotiva, arquitetura e até em efeitos visuais para cinema e jogos. Resolver as equações de Navier-Stokes de forma direta é caro computacionalmente e historicamente restrito a simulações offline rodando em clusters.

A escolha deste projeto nasce de um tipo de demonstração interativa bastante popular em sites de divulgação científica e em ferramentas educacionais: um "túnel de vento virtual", onde o usuário desenha ou escolhe um obstáculo e observa, em tempo real, como o fluxo de ar (visualizado como fumaça/linhas de corrente) se comporta ao redor dele. Esse tipo de demonstração só é viável em tempo real graças à paralelização massiva em GPU — é exatamente esse contraste (impossível em tempo real na CPU, trivial na GPU) que motiva a escolha do tema para a disciplina.

Em vez de resolver Navier-Stokes diretamente, este projeto usa o **Lattice Boltzmann Method (LBM)**, um método numérico alternativo que modela o fluido em um nível mesoscópico (pacotes de partículas se movendo em um reticulado) em vez de resolver as equações diferenciais parciais diretamente. O LBM é particularmente atraente para este projeto porque:

- é significativamente mais simples de implementar do que um solver completo de Navier-Stokes (como o método *Stable Fluids* usado no sample `fluidsGL` da NVIDIA, baseado em projeção de pressão via FFT);
- lida de forma natural com obstáculos de geometria arbitrária, bastando marcar células da grade como "sólidas" — sem precisar de malhas de contorno complexas (*boundary-fitted meshes*);
- é, por construção, um algoritmo **local e baseado em grade regular**, o que o torna um dos métodos de CFD mais usados em GPU na literatura.

Além do apelo técnico, o projeto propõe uma extensão para **3D**, com uma cena no estilo *viewport* de ferramentas de modelagem (como o Blender), onde o usuário escolhe uma forma base (esfera, cubo, cilindro, toroide, boneco-palito) e a deforma de maneira simplificada antes de "jogá-la" no túnel de vento. Isso mantém o projeto visualmente ambicioso e pessoalmente motivador, sem exigir a reconstrução de um motor de escultura de malhas (o que seria um projeto à parte, não relacionado a paralelismo).

## 2. Descrição do algoritmo

### 2.1 Fundamentos do Lattice Boltzmann Method

O LBM discretiza o domínio do fluido em uma grade regular de células. Em cada célula, em vez de armazenar diretamente densidade e velocidade, o método guarda um conjunto de **funções de distribuição** `f_i`, que representam a fração de "pacotes" de fluido se movendo em um pequeno conjunto fixo de direções discretas. No caso 3D, o modelo mais comum é o **D3Q19** (3 dimensões, 19 direções discretas por célula), que equilibra bem precisão física e custo computacional.

A cada passo de tempo, o algoritmo executa duas etapas por célula:

1. **Colisão (*collision*)**: as funções de distribuição são relaxadas em direção a um estado de equilíbrio local (calculado a partir da densidade e velocidade macroscópicas daquela célula), usando o operador BGK (*Bhatnagar-Gross-Krook*). Essa etapa modela a "troca de momento" entre partículas.
2. **Propagação (*streaming*)**: cada `f_i` é deslocada para a célula vizinha correspondente à sua direção de movimento.

As grandezas macroscópicas (densidade `ρ` e velocidade `u`) são recuperadas a cada passo como momentos estatísticos simples das `f_i` armazenadas naquela célula.

As **condições de contorno** são o que torna o método especialmente adequado a este projeto:
- **Obstáculo sólido**: regra de *bounce-back* — qualquer `f_i` que atinja uma célula marcada como sólida simplesmente é refletida na direção oposta, sem necessidade de cálculos especiais de geometria;
- **Entrada de vento**: uma condição de velocidade constante é imposta em uma face da caixa de simulação;
- **Saída**: condição de fluxo livre na face oposta.

### 2.2 Da forma 3D ao obstáculo da simulação

A forma escolhida/deformada pelo usuário (malha de triângulos simples, sem topologia dinâmica) é convertida em um conjunto de células "sólidas" dentro da grade 3D através de um passo de **voxelização** — executado apenas quando o usuário edita a forma, não a cada frame, já que o custo desse passo é bem menor que o da simulação contínua.

### 2.3 Por que esse problema tem alta carga computacional

Para uma simulação em tempo real (24–60 FPS) com uma grade de, por exemplo, 128×128×128 células (~2 milhões de células), cada frame exige recalcular colisão e propagação para todas as células, cada uma envolvendo um número não trivial de operações de ponto flutuante sobre as 19 direções do D3Q19. Isso representa dezenas a centenas de milhões de operações por frame, sustentadas continuamente enquanto a simulação roda — uma carga claramente inviável em tempo real para uma implementação sequencial, mas perfeitamente tratável em GPU.

### 2.4 Benefícios da paralelização

A atualização de cada célula do LBM depende **apenas do estado da própria célula e de suas vizinhas imediatas** no passo anterior (um padrão de *stencil*, semelhante ao Jogo da Vida ou à equação do calor vistos em aula). Não há dependências globais dentro de um mesmo passo de tempo, o que torna o algoritmo **embaraçosamente paralelo** por célula — um encaixe quase perfeito com o paradigma SIMT das GPUs (threads independentes executando a mesma instrução sobre dados diferentes).

## 3. Arquitetura paralela

O projeto compara três implementações do mesmo algoritmo, isolando o efeito da arquitetura paralela sobre o desempenho:

- **Serial (baseline)**: implementação sequencial em C/C++, usada como referência para o cálculo de speedup.
- **OpenMP (CPU multi-thread)**: paralelização do laço sobre as células da grade com `#pragma omp parallel for`, testando diferentes números de threads e políticas de `schedule`, conforme visto em aula.
- **CUDA (GPU)**: cada célula (ou pequeno bloco de células) é mapeada para uma thread CUDA; o kernel implementa colisão e propagação, usando memória compartilhada (*shared memory*) para reduzir acessos à memória global ao reutilizar dados de vizinhança dentro de um bloco de threads.

A **GPU é a arquitetura-alvo principal** do projeto, pela natureza do problema: o trabalho por célula é relativamente leve (poucas dezenas de operações), mas o número de células é muito grande e todas são independentes — exatamente o cenário em que a arquitetura orientada a vazão (*throughput*) da GPU, com milhares de núcleos simples, supera a arquitetura orientada a latência da CPU. O comparativo com OpenMP serve para demonstrar, de forma concreta, os limites do paralelismo em CPU (poucos núcleos, mesmo que eficientes individualmente) frente ao paralelismo massivo da GPU.

Paralelismo a nível de múltiplas máquinas (MPI) está fora do escopo deste projeto, já que o objetivo é uma aplicação interativa de estação de trabalho única — mas é mencionado como trabalho futuro para simulações offline de maior escala (LBM distribuído entre múltiplos nós/GPUs).

## 4. Tecnologia

- **Linguagem**: C/C++.
- **Paralelismo CPU**: OpenMP.
- **Paralelismo GPU**: CUDA (kernels de colisão + propagação, uso de memória compartilhada).
- **Visualização/interação 3D**: biblioteca gráfica leve (ex.: raylib ou OpenGL/GLFW) para o *viewport* 3D com câmera orbital, menu de formas base e deformação simplificada por "pincel" (deslocamento de vértices ao longo da normal, sem alteração de topologia).
- **Visualização do fluido**: corte por plano (*slice*) da grade de densidade/velocidade e, se o tempo permitir, *ray marching* sobre a grade de densidade para um efeito de fumaça volumétrica.
- **Ferramentas de análise de desempenho**: NVIDIA Nsight Compute / `nvprof` para profiling dos kernels CUDA; medição de tempo por frame (ms) e FPS nas três versões.

## 5. Resultados esperados

**Resultados visuais:**
- Capturas de tela e vídeo da simulação interativa, mostrando o fluxo (linhas de corrente/fumaça) contornando diferentes obstáculos (formas base e variações deformadas) na caixa de vento 3D.

**Resultados de desempenho:**
- Tabela/gráfico comparando tempo de execução por frame (ms) e FPS atingido entre Serial, OpenMP (variando número de threads) e CUDA, para diferentes resoluções de grade (ex.: 32³, 64³, 128³ células).
- Gráfico de *speedup* (OpenMP/Serial e CUDA/Serial) em função do tamanho do problema (resolução da grade).
- Discussão sobre o limiar de resolução/carga a partir do qual a interatividade em tempo real (≥ 24–30 FPS) só é alcançável com a versão CUDA.

## 6. Referências

1. CHEN, S.; DOOLEN, G. D. **Lattice Boltzmann method for fluid flows**. Annual Review of Fluid Mechanics, v. 30, p. 329–364, 1998.
2. SUCCI, S. **The Lattice Boltzmann Equation for Fluid Dynamics and Beyond**. Oxford University Press, 2001.
3. TÖLKE, J.; KRAFCZYK, M. **TeraFLOP computing on a desktop PC with GPUs for 3D CFD**. International Journal of Computational Fluid Dynamics, v. 22, n. 7, p. 443–456, 2008.
4. OBRECHT, C. et al. **Multi-GPU implementation of the lattice Boltzmann method**. Computers & Mathematics with Applications, v. 65, n. 2, p. 252–261, 2013.
5. HARRIS, M. **Fast Fluid Dynamics Simulation on the GPU**. In: GPU Gems, cap. 38, NVIDIA/Addison-Wesley, 2004. (técnica alternativa baseada em Navier-Stokes, usada como contraponto ao LBM)
6. BRIDSON, R. **Fluid Simulation for Computer Graphics**. 2. ed. CRC Press, 2015.
7. NVIDIA. **CUDA Samples — fluidsGL e particles**. Disponível em: https://github.com/NVIDIA/cuda-samples (referência de implementação e técnicas de interação/visualização em GPU).
