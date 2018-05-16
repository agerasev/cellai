// Constants (will be substituted)

#define WORLD_SIZE_X
#define WORLD_SIZE_Y
#define SIZE_A_AGENT_I
#define SIZE_A_AGENT_F
#define SIZE_W_CACHE_I
#define SIZE_W_CACHE_F
#define SIZE_W_TRACE_F
#define SIZE_W_OBJECT_I
#define SIZE_A_RNN_F
#define RNN_SIZE_X
#define RNN_SIZE_H
#define RNN_SIZE_Y


// Parameters location

#define F_TRACE_FADE_FAC        0
#define F_TRACE_DIFF_FAC        1
#define F_TRACE_ANIMAL_FAC      2
#define F_TRACE_PLANT_FAC       3
#define F_PLANT_APPEAR_PROB     4


// Object types

#define OBJECT_NONE             0x00
#define OBJECT_PLANT            0x01


// Structures

typedef struct {
    int2 pos;
    int score;
    __global float *memory;
} Agent;

Agent agent_load(
    int index,
    __global int   *a_agent_i, 
    __global float *a_agent_f
) {
    __global int   *agent_i = a_agent_i + index*SIZE_A_AGENT_I;
    __global float *agent_f = a_agent_f + index*SIZE_A_AGENT_F;
    Agent a;
    a.pos = (int2)(agent_i[0], agent_i[1]);
    a.score = agent_i[2];
    a.memory = agent_f;
    return a;
}

void agent_store(
    Agent a,
    int index,
    __global int   *a_agent_i, 
    __global float *a_agent_f
) {
    __global int   *agent_i = a_agent_i + index*SIZE_A_AGENT_I;
    __global float *agent_f = a_agent_f + index*SIZE_A_AGENT_F;
    agent_i[0] = a.pos.x;
    agent_i[1] = a.pos.y;
    agent_i[2] = a.score;
}

typedef struct {
    __global float *Wxh;
    __global float *bh;
    __global float *Whh;
    __global float *Why;
    __global float *by;
} RNN;

RNN rnn_reference(
    int index,
    __global float *a_rnn_f
) {
    RNN rnn;
    __global float *rnn_ptr = a_rnn_f + index*SIZE_A_RNN_F;
    rnn.Wxh = rnn_ptr;
    rnn_ptr += RNN_SIZE_X*RNN_SIZE_H;
    rnn.bh = rnn_ptr;
    rnn_ptr += RNN_SIZE_H;
    rnn.Whh = rnn_ptr;
    rnn_ptr += RNN_SIZE_H*RNN_SIZE_H;
    rnn.Why = rnn_ptr;
    rnn_ptr += RNN_SIZE_H*RNN_SIZE_Y;
    rnn.by = rnn_ptr;
    return rnn;
}

void rnn_step(RNN rnn, float *x, __global float *h, float *y) {
    int i, j;
    float t;
    float ht[RNN_SIZE_H];
    for (i = 0; i < RNN_SIZE_H; ++i) {
        t = 0.0;
        for (j = 0; j < RNN_SIZE_H; ++j) {
            t += h[j]*rnn.Whh[i*RNN_SIZE_H + j];
        }
        ht[i] = t;
        
        t = 0;
        for (j = 0; j < RNN_SIZE_X; ++j) {
            t += x[j]*rnn.Wxh[i*RNN_SIZE_X + j];
        }
        ht[i] += t;
        
        ht[i] += rnn.bh[i];
        
        ht[i] = tanh(ht[i]);
    }
    
    for (i = 0; i < RNN_SIZE_H; ++i) {
        h[i] = ht[i];
    }
    
    for (i = 0; i < RNN_SIZE_Y; ++i) {
        t = 0;
        for (j = 0; j < RNN_SIZE_H; ++j) {
            t += ht[j]*rnn.Why[i*RNN_SIZE_H + j];
        }
        y[i] = t + rnn.by[i];
    }
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
    __constant int   *P_i,
    __constant float *P_f,
    
    __global uint *w_random,
    
    __global int   *w_cache_i,
    __global float *w_cache_f,
    
    __global const int *w_object_i,
    __global const float *w_trace_f
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    
    // diffusion
    float3 v = vload3(0, w_trace_f + fpos(gi, gs)*SIZE_W_TRACE_F);
    float3 dv = 0.0;
    int np[4];
    int i;
    for (i = 0; i < 4; ++i) {
        int2 ni = gi + idir(i);
        float3 nv;
        if (is_inside(ni, (int2)(0,0), gs)) {
            nv = vload3(0, w_trace_f + fpos(ni, gs)*SIZE_W_TRACE_F);
        } else {
            nv = v;
        }
        dv += P_f[F_TRACE_DIFF_FAC]*(nv - v);
    }
    vstore3(dv, 0, w_cache_f + fpos(gi, gs)*SIZE_W_CACHE_F);
}

__kernel void w_step_write(
    __constant int   *P_i,
    __constant float *P_f,
    
    __global uint *w_random,
    
    __global int   *w_cache_i,
    __global float *w_cache_f,
    
    __global int *w_object_i,
    __global float *w_trace_f
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    
    int fp = fpos(gi, gs);
    
    // fade trace
    float3 v = vload3(0, w_trace_f + fp*SIZE_W_TRACE_F);
    float3 dv = vload3(0, w_cache_f + fp*SIZE_W_CACHE_F);
    v += dv;
    v *= 1.0f - P_f[F_TRACE_FADE_FAC];
    vstore3(v, 0, w_trace_f + fp*SIZE_W_TRACE_F);
    
    // plants
    if (rand_uniform(&(w_random[fp])) < P_f[F_PLANT_APPEAR_PROB]) {
        *(w_object_i + fp*SIZE_W_OBJECT_I) = OBJECT_PLANT;
    }
    
    if (*(w_object_i + fp*SIZE_W_OBJECT_I) == OBJECT_PLANT) {
        v.y = 1.0f - (1.0f - P_f[F_TRACE_PLANT_FAC])*(1.0f - v.y);
        vstore3(v, 0, w_trace_f + fp*SIZE_W_TRACE_F);
    }
}


__kernel void a_step(
    __constant int   *P_i,
    __constant float *P_f,
    
    __global uint *a_random,

    __global int   *a_agent_i,
    __global float *a_agent_f,
    __global float *a_rnn_f,
    
    __global int *w_object_i,
    __global float *w_trace_f
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(WORLD_SIZE_X, WORLD_SIZE_Y);

    Agent agent = agent_load(gi, a_agent_i, a_agent_f);
    RNN rnn = rnn_reference(gi, a_rnn_f);
    
    int fp = (agent.pos.x + ws.x*agent.pos.y);
    float3 v = vload3(0, w_trace_f + fp*SIZE_W_TRACE_F);
    int i;
    float x[RNN_SIZE_X];
    float y[RNN_SIZE_Y];
    x[0] = v.y;
    for (i = 0; i < 4; ++i) {
        int2 ni = gi + idir(i);
        float3 nv;
        if (is_inside(ni, (int2)(0,0), ws)) {
            nv = vload3(0, w_trace_f + fpos(ni, ws)*SIZE_W_TRACE_F);
        } else {
            nv = v;
        }
        x[i+1] = v.y;
    }
    rnn_step(rnn, x, agent.memory, y);
    // softmax
    float my = y[0];
    for (i = 0; i < RNN_SIZE_Y - 1; ++i) {
        my = max(my, y[i+1]);
    }
    float sy = 0.0;
    for (i = 0; i < RNN_SIZE_Y; ++i) {
        y[i] -= my;
        y[i] = exp(y[i]);
        sy += y[i];
    }
    float cy = 0.0;
    float ry = rand_uniform(a_random + gi);
    int d = 0;
    for (i = 0; i < RNN_SIZE_Y; ++i) {
        y[i] /= sy;
        cy += y[i];
        if (ry > cy) {
            d = i+1;
        }
    }
    
    agent.pos += (d != 0)*(int2)((d%2)*(2-d), ((d+1)%2)*(3-d));
    agent.pos = clamp(agent.pos, (int2)(0, 0), ws - (int2)(1,1));
    agent_store(agent, gi, a_agent_i, a_agent_f);

    v.x = 1.0f - (1.0f - P_f[F_TRACE_ANIMAL_FAC])*(1.0f - v.x);
    vstore3(v, 0, w_trace_f + fp*SIZE_W_TRACE_F);
}


__kernel void w_draw(
    __constant int   *P_i,
    __constant float *P_f,
    
    __global const int *w_object_i,
    __global const float *w_trace_f,
    __global uchar *w_screen
) {
    int2 gi = (int2)(get_global_id(0), get_global_id(1));
    int2 gs = (int2)(get_global_size(0), get_global_size(1));
    int fp = (gi.x + gs.x*gi.y);
    uchar3 col = convert_uchar3(255.0f*clamp(vload3(0, w_trace_f + fp*SIZE_W_TRACE_F), 0.0f, 1.0f));
    vstore3(col, fp, w_screen);
    
    int obj = *(w_object_i + fp*SIZE_W_OBJECT_I);
    if (obj != OBJECT_NONE) {
        uchar3 col;
        if (obj == OBJECT_PLANT) {
            col = (uchar3)(0,255,0);
        } else {
            col = (uchar3)(255,255,255);
        }
        vstore3(col, fp, w_screen);
    }
}


__kernel void a_draw(
    __constant int   *P_i,
    __constant float *P_f,

    __global const int *a_agent_i,
    __global const int *a_agent_f,
    
    __global uchar *w_screen
) {
    int gi = get_global_id(0);
    int gs = get_global_size(0);
    int2 ws = (int2)(WORLD_SIZE_X, WORLD_SIZE_Y);

    Agent agent = agent_load(gi, a_agent_i, a_agent_f);
    int p = (agent.pos.x + ws.x*agent.pos.y);
    uchar3 col = (uchar3)(255,0,0);
    vstore3(col, p, w_screen);
}
