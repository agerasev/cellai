// Parameters location

#define I_WORLD_SIZE_X          0
#define I_WORLD_SIZE_Y          1
#define I_AGENT_SIZE_I          2
#define I_AGENT_SIZE_F          3

#define F_TR_FADE_FAC           0
#define F_TR_DIFF_FAC           1


// Structures

typedef struct {
    int2 pos;
    int score;
} Agent;

Agent agent_load(
    __global const int   *agent_i, 
    __global const float *agent_f
) {
    Agent a;
    a.pos = (int2)(agent_i[0], agent_i[1]);
    a.score = agent_i[2];
    return a;
}

Agent agent_store(
    Agent a,
    __global int   *agent_i, 
    __global float *agent_f
) {
    agent_i[0] = a.pos.x;
    agent_i[1] = a.pos.y;
    agent_i[2] = a.score;
}

// Functions

uint rand_int(__global uint *seed) {
    return (*seed = 1103515245**seed + 12345);
}

float rand_uniform(__global uint *seed) {
    return (float)rand_int(seed)/(float)0xffffffff;
}


bool is_inside(int2 p, int2 l, int2 h) {
    return l.x <= p.x && p.x < h.x && l.y <= p.y && p.y < h.y;
}

int2 idir(int idx) {
    return (int2)(((idx+1)%2)*(1-idx), (idx%2)*(2-idx));
}

int fpos(int2 pos, int2 size) {
    return pos.x + size.x*pos.y;
}


// Kernels

__kernel void w_step(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global uint *w_random,
    
    __global const float *w_trace__src,
    __global       float *w_trace__dst
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    float v = w_trace__src[fpos(gi, gs)];
    
    // diffusion
    float dv = 0.0;
    int np[4];
    int i;
    for (i = 0; i < 4; ++i) {
        int2 ni = gi + idir(i);
        float nv;
        if (is_inside(ni, (int2)(0,0), gs)) {
            nv = w_trace__src[fpos(ni, gs)];
        } else {
            nv = v;
        }
        dv += PAR_F[F_TR_DIFF_FAC]*(nv - v);
    }
    v += dv;
    
    // fade
    v *= 1.0 - PAR_F[F_TR_FADE_FAC];
    
    w_trace__dst[fpos(gi, gs)] = v;
}


__kernel void a_step(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global uint *a_random,

    __global int   *a_agents_i,
    __global float *a_agents_f,
    
    __global const float *w_trace__src,
    __global       float *w_trace__dst
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(PAR_I[I_WORLD_SIZE_X], PAR_I[I_WORLD_SIZE_Y]);
    __global int *agent_i = a_agents_i + gi*PAR_I[I_AGENT_SIZE_I];

    int2 pos = vload2(0, agent_i);
    int d = rand_int(&(a_random[gi])) % 5;
    pos += (d != 0)*(int2)((d%2)*(2-d), ((d+1)%2)*(3-d));
    pos = clamp(pos, (int2)(0, 0), ws - (int2)(1,1));
    vstore2(pos, 0, agent_i);

    uint p = (pos.x + ws.x*pos.y);
    w_trace__dst[p] = 1.0;
}


__kernel void w_draw(
    __constant int *PAR_I,
    __constant float *PAR_F,
    
    __global const float *w_trace,
    __global uchar *w_screen
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    int p = (gi.x + gs.x*gi.y);
    uchar c = (uchar)(255*w_trace[p]);
    uchar3 col = (uchar3)(c, c, 0);
    vstore3(col, p, w_screen);
}


__kernel void a_draw(
    __constant int *PAR_I,
    __constant float *PAR_F,

    __global const int *a_agents_i,
    __global const int *a_agents_f,
    
    __global uchar *w_screen
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(PAR_I[I_WORLD_SIZE_X], PAR_I[I_WORLD_SIZE_Y]);
    const __global int *agent_i = a_agents_i + gi*PAR_I[I_AGENT_SIZE_I];

    int2 pos = vload2(0, agent_i);
    int p = (pos.x + ws.x*pos.y);
    uchar c = 255;
    uchar3 col = (uchar3)(c, c, c);
    vstore3(col, p, w_screen);
}
