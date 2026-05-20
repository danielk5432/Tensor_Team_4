# Geometry of Tensors and Applications

## Team 4

### Targeting Problem : 5 — Low Rank Tensor Approximation

Develop a method of approximating any given tensor in $\mathbb{C}^3 \otimes \mathbb{C}^3 \otimes \mathbb{C}^3$ (i.e. Rubik's cube case) by rank 2 tensor decomposition. In other words, for any given tensor $T \in \mathbb{C}^3 \otimes \mathbb{C}^3 \otimes \mathbb{C}^3$, find a best approximate rank 2 tensor 

$$T' \in \sigma_2^0(\mathrm{Seg}(\mathbb{C}^3 \times \mathbb{C}^3 \times \mathbb{C}^3))$$

such that the distance $\|T - T'\|$ is minimal (of course, first you need to decide which distance function you will use). Also, implement it by computer code..

## 요약

- $\mathbb{C}^3 \otimes \mathbb{C}^3 \otimes \mathbb{C}^3$ 안의 임의의 텐서를 rank 2로 근사하는 방법을 개발하고 코드로 구현.
- Distance function 선택은 자유 — 어떤 norm을 쓸지 먼저 결정해야 함.
- 목표: $\|T - T'\|$를 최소화하는 $T' \in \sigma_2^0(\mathrm{Seg}(\mathbb{C}^3 \times \mathbb{C}^3 \times \mathbb{C}^3))$ 찾기.

팀원:
김동현
박사빈
정범준
조수호

---

## 참고

- [Git 기초 튜토리얼](documentation/git_tutorial.md) — Git을 처음 사용하는 팀원을 위한 가이드