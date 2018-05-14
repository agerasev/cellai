// Parameters location

#define I_WORLD_SIZE_X          0
#define I_WORLD_SIZE_Y          1
#define I_AGENT_SIZE_I          2
#define I_AGENT_SIZE_F          3
#define I_W_CACHE_SIZE_I        4
#define I_W_CACHE_SIZE_F        5
#define I_W_TRACE_SIZE_F        6

#define F_TRACE_FADE_FAC        0
#define F_TRACE_DIFF_FAC        1
#define F_TRACE_APPEAR_FAC      2

// Structures

typedef struct {
    int2 pos;
    int score;
} Agent;

Agent agent_load(
    __constant int *PAR_I,
    int index,
    __global const int   *a_agents_i, 
    __global const float *a_agents_f
) {
    __global const int   *agent_i = a_agents_i + PAR_I[I_AGENT_SIZE_I]*index;
    __global const float *agent_f = a_agents_f + PAR_I[I_AGENT_SIZE_F]*index;
    Agent a;
    a.pos = (int2)(agent_i[0], agent_i[1]);
    a.score = agent_i[2];
    return a;
}

void agent_store(
    Agent a,
    __constant int *PAR_I,
    int index,
    __global int   *a_agents_i, 
    __global float *a_agents_f
) {
    __global int   *agent_i = a_agents_i + PAR_I[I_AGENT_SIZE_I]*index;
    __global float *agent_f = a_agents_f + PAR_I[I_AGENT_SIZE_F]*index;
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

__kernel void w_step_read(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global uint *w_random,
    
    __global int   *w_cache_i,
    __global float *w_cache_f,
    
    __global const float *w_trace_f
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    
    // diffusion
    float3 v = vload3(0, w_trace_f + fpos(gi, gs)*PAR_I[I_W_TRACE_SIZE_F]);
    float3 dv = 0.0;
    int np[4];
    int i;
    for (i = 0; i < 4; ++i) {
        int2 ni = gi + idir(i);
        float3 nv;
        if (is_inside(ni, (int2)(0,0), gs)) {
            nv = vload3(0, w_trace_f + fpos(ni, gs)*PAR_I[I_W_TRACE_SIZE_F]);
        } else {
            nv = v;
        }
        dv += PAR_F[F_TRACE_DIFF_FAC]*(nv - v);
    }
    vstore3(dv, 0, w_cache_f + fpos(gi, gs)*PAR_I[I_W_CACHE_SIZE_F]);
}

__kernel void w_step_write(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global uint *w_random,
    
    __global int   *w_cache_i,
    __global float *w_cache_f,
    
    __global float *w_trace_f
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    
    int fp = fpos(gi, gs);
    
    // fade
    float3 v = vload3(0, w_trace_f + fp*PAR_I[I_W_TRACE_SIZE_F]);
    float3 dv = vload3(0, w_cache_f + fp*PAR_I[I_W_CACHE_SIZE_F]);
    v += dv;
    v *= 1.0f - PAR_F[F_TRACE_FADE_FAC];
    vstore3(v, 0, w_trace_f + fp*PAR_I[I_W_TRACE_SIZE_F]);
}


__kernel void a_step(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global uint *a_random,

    __global int   *a_agents_i,
    __global float *a_agents_f,
    
    __global float *w_trace_f
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(PAR_I[I_WORLD_SIZE_X], PAR_I[I_WORLD_SIZE_Y]);
    __global int *agent_i = a_agents_i + gi*PAR_I[I_AGENT_SIZE_I];

    Agent agent = agent_load(PAR_I, gi, a_agents_i, a_agents_f);
    int d = rand_int(&(a_random[gi])) % 5;
    agent.pos += (d != 0)*(int2)((d%2)*(2-d), ((d+1)%2)*(3-d));
    agent.pos = clamp(agent.pos, (int2)(0, 0), ws - (int2)(1,1));
    agent_store(agent, PAR_I, gi, a_agents_i, a_agents_f);

    int fp = (agent.pos.x + ws.x*agent.pos.y);
    float3 v = vload3(0, w_trace_f + fp*PAR_I[I_W_TRACE_SIZE_F]);
    v.y = 1.0f - (1.0f - PAR_F[F_TRACE_APPEAR_FAC])*(1.0f - v.y);
    vstore3(v, 0, w_trace_f + fp*PAR_I[I_W_TRACE_SIZE_F]);
}


__kernel void w_draw(
    __constant int   *PAR_I,
    __constant float *PAR_F,
    
    __global const float *w_trace_f,
    __global uchar *w_screen
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    int fp = (gi.x + gs.x*gi.y);
    uchar3 col = convert_uchar3(255.0f*vload3(0, w_trace_f + fp*PAR_I[I_W_TRACE_SIZE_F]));
    vstore3(col, fp, w_screen);
}


__kernel void a_draw(
    __constant int   *PAR_I,
    __constant float *PAR_F,

    __global const int *a_agents_i,
    __global const int *a_agents_f,
    
    __global uchar *w_screen
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(PAR_I[I_WORLD_SIZE_X], PAR_I[I_WORLD_SIZE_Y]);
    const __global int *agent_i = a_agents_i + gi*PAR_I[I_AGENT_SIZE_I];

    Agent agent = agent_load(PAR_I, gi, a_agents_i, a_agents_f);
    int p = (agent.pos.x + ws.x*agent.pos.y);
    uchar c = 255;
    uchar3 col = (uchar3)(c, c, c);
    vstore3(col, p, w_screen);
}
