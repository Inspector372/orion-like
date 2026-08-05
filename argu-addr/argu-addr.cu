/*
    argu-addr.cu

    kernel function parameter의 'param memory' 주소를 읽어올 수 있음
    (The address of a kernel parameter may be moved into a register using the mov instruction.
    The resulting address is in the .param state space and is accessed using ld.param instructions.)

    .reg .u32 %n;
    ld.param.u32 %n, [param];

    첫 번째 parameter의 주소는 CONSTANT_BUFFER_ADDR_LOWER/UPPER(i)와 0x168만큼의 offset 차이가 남

    Kernel(i)를 Kernel launch의 i번째 atom이라 하자.
    1. QMD에서 CONSTANT_BUFFER_ADDR_LOWER/UPPER(0), PROGRAM_ADDRESS를 미리 fetch -> BufferAddr(i), ProgramAddr
    2. Global Memory에 Map[(Addr(i) +(or -) 0x168) -> (i, ProgramAddr)]을 저장 (Hash Table or something?)
    3. 실제 Kernel 실행에서, wrapper(int a); 형식에서 &a를 inline ptx로 읽음
    4. Map(&a) = (i, ProgramAddr)
    5. i는 index filtering에 사용, ProgramAddr은 jump에 사용

*/

