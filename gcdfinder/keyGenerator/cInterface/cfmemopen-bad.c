/*-
 * Copyright (c) 2004-2005
 *	Bruce Korb.  All rights reserved.
 *
 * Time-stamp:      "2005-03-08 20:56:57 bkorb"
 *
 * This code was inspired from software written by
 *   Hanno Mueller, kontakt@hanno.de
 * and completely rewritten by Bruce Korb, bkorb@gnu.org
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the author nor the name of any other contributor
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#if defined(ENABLE_FMEMOPEN)

/*=--subblock=arg=arg_type,arg_name,arg_desc =*/
/*=*
 * library:  fmem
 * header:   libfmem.h
 *
 * lib_description:
 *
 *  This library only functions in the presence of GNU or BSD's libc.
 *  It is distributed under the Berkeley Software Distribution License.
 *  This library can only function if there is a libc-supplied mechanism
 *  for fread/fwrite/fseek/fclose to call into this code.  GNU uses
 *  fopencookie and BSD supplies funopen.
=*/
/*
 * fmemopen() - "my" version of a string stream
 * inspired by the glibc version, but completely rewritten and
 * extended by Bruce Korb - bkorb@gnu.org
 *
 * - See the man page.
 */

#if defined(__linux) || defined(__linux__)
#  define _GNU_SOURCE
#endif

#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(HAVE_FOPENCOOKIE)
#  include <libio.h>
   typedef _IO_off64_t  fmem_off_t;
   typedef int          seek_pos_t;

#elif defined(HAVE_FUNOPEN)
   typedef size_t  fmem_off_t;
   typedef fpos_t  seek_pos_t;

   typedef int     (cookie_read_function_t )(void *, char *, int);
   typedef int     (cookie_write_function_t)(void *, const char *, int);
   typedef fpos_t  (cookie_seek_function_t )(void *, fpos_t, int);
   typedef int     (cookie_close_function_t)(void *);

#endif

#define PROP_TABLE \
_Prop_( read,       "Read from buffer"        ) \
_Prop_( write,      "Write to buffer"         ) \
_Prop_( append,     "Append to buffer okay"   ) \
_Prop_( binary,     "byte data - not string"  ) \
_Prop_( create,     "allocate the string"     ) \
_Prop_( truncate,   "start writing at start"  ) \
_Prop_( allocated,  "we allocated the buffer" )

#define _Prop_(n,s)   BIT_ID_ ## n,
typedef enum { PROP_TABLE BIT_CT } fmem_flags_e;
#undef  _Prop_

#define FLAG_BIT(n)  (1 << BIT_ID_ ## n)

typedef unsigned long mode_bits_t;
typedef unsigned char buf_chars_t;

typedef struct fmem_cookie_s fmem_cookie_t;
struct fmem_cookie_s {
    mode_bits_t    mode;
    buf_chars_t   *buffer;
    size_t         buf_size;    /* Full size of buffer */
    size_t         next_ix;     /* Current position */
    size_t         high_water;  /* Highwater mark of valid data */
    size_t         pg_size;     /* size of a memory page.
                                   Future architectures allow it to vary
                                   by memory region. */
};

/* = = = START-STATIC-FORWARD = = = */
/* static forward declarations maintained by :mkfwd */
static int
fmem_getmode( const char *pMode, mode_bits_t *pRes );

static int
fmem_extend( fmem_cookie_t *pFMC, size_t new_size );

static ssize_t
fmem_read( void *cookie, void *pBuf, size_t sz );

static ssize_t
fmem_write( void *cookie, const void *pBuf, size_t sz );

static seek_pos_t
fmem_seek (void *cookie, fmem_off_t *p_offset, int dir);

static int
fmem_close( void *cookie );
/* = = = END-STATIC-FORWARD = = = */

#ifdef TEST_FMEMOPEN
  static fmem_cookie_t* saved_cookie = NULL;
#endif

static int
fmem_getmode( const char *pMode, mode_bits_t *pRes )
{
    if (pMode == NULL)
        return 1;

    switch (*pMode) {
    case 'a': *pRes = FLAG_BIT(write) | FLAG_BIT(create) | FLAG_BIT(append);
              break;
    case 'w': *pRes = FLAG_BIT(write) | FLAG_BIT(create) | FLAG_BIT(truncate);
              break;
    case 'r': *pRes = FLAG_BIT(read);
              break;
    default:  return EINVAL;
    }

    /*
     *  If someone wants to supply a "wxxbxbbxbb+" pMode string, I don't care.
     */
    for (;;) {
        switch (*++pMode) {
        case '+': *pRes |= FLAG_BIT(read) | FLAG_BIT(write);
                  if (pMode[1] != NUL)
                      return EINVAL;
                  break;
        case NUL: break;
        case 'b': *pRes |= FLAG_BIT(binary); continue;
        case 'x': continue;
        default:  return EINVAL;
        }
        break;
    }

    return 0;
}

static int
fmem_extend( fmem_cookie_t *pFMC, size_t new_size )
{
    size_t ns = (new_size + (pFMC->pg_size - 1)) & (~(pFMC->pg_size - 1));

#   define APPEND_OK_MASK (FLAG_BIT(write) | FLAG_BIT(append))
    /*
     *  We can expand the buffer only if we are in append mode.
     */
    if ((pFMC->mode & FLAG_BIT(append)) == 0)
        goto no_space;

    if ((pFMC->mode & FLAG_BIT(allocated)) == 0) {
        /*
         *  Previously, this was a user supplied buffer.  We now move to one
         *  of our own.  The user is responsible for the earlier memory.
         */
        void* bf = malloc( ns );
        if (bf == NULL)
            goto no_space;

        memcpy( bf, pFMC->buffer, pFMC->buf_size );
        pFMC->buffer = bf;
        pFMC->mode  |= FLAG_BIT(allocated);
    }
    else {
        void* bf = realloc( pFMC->buffer, ns );
        if (bf == NULL)
            goto no_space;

        pFMC->buffer = bf;
    }

    /*
     *  Unallocated file space is set to zeros.  Emulate that.
     */
    memset( pFMC->buffer + pFMC->buf_size, 0, ns - pFMC->buf_size );
    pFMC->buf_size = ns;
    return 0;

 no_space:
    errno = ENOSPC;
    return -1;
}

static ssize_t
fmem_read( void *cookie, void *pBuf, size_t sz )
{
    fmem_cookie_t *pFMC = cookie;

    if (pFMC->next_ix + sz > pFMC->buf_size) {
        if (pFMC->next_ix >= pFMC->buf_size)
            return (sz > 0) ? -1 : 0;
        sz = pFMC->buf_size - pFMC->next_ix;
    }

    memcpy( pBuf, pFMC->buffer + pFMC->next_ix, sz );

    pFMC->next_ix += sz;
    if (pFMC->next_ix > pFMC->high_water)
        pFMC->high_water = pFMC->next_ix;

    return sz;
}


static ssize_t
fmem_write( void *cookie, const void *pBuf, size_t sz )
{
    fmem_cookie_t *pFMC = cookie;
    int add_nul_char;

    /*
     *  In append mode, always seek to the end before writing.
     */
    if (pFMC->mode & FLAG_BIT(append))
        pFMC->next_ix = pFMC->high_water;

    /*
     *  Only add a NUL character if:
     *
     *  * we are not in binary mode
     *  * there are data to write
     *  * the last character to write is not already NUL
     */
    add_nul_char =
           ((pFMC->mode & FLAG_BIT(binary)) != 0)
        && (sz > 0)
        && (((char*)pBuf)[sz - 1] != NUL);

    {
        size_t next_pos = pFMC->next_ix + sz + add_nul_char;
        if (next_pos > pFMC->buf_size) {
            if (fmem_extend( pFMC, next_pos ) != 0) {
                /*
                 *  We could not extend the memory.  Try to write some data.
                 *  Fail if we are either at the end or not writing data.
                 */
                if ((pFMC->next_ix >= pFMC->buf_size) || (sz == 0))
                    return -1; /* no space at all.  errno is set. */

                /*
                 *  Never add the NUL for a truncated write.  "sz" may be
                 *  unchanged or limited here.
                 */
                add_nul_char = 0;
                sz = pFMC->buf_size - pFMC->next_ix;
            }
        }
    }

    memcpy( pFMC->buffer + pFMC->next_ix, pBuf, sz);

    pFMC->next_ix += sz;

    /*
     *  Check for new high water mark and remember it.  Add a NUL if
     *  we do that and if we have a new high water mark.
     */
    if (pFMC->next_ix > pFMC->high_water) {
        pFMC->high_water = pFMC->next_ix;
        if (add_nul_char)
            /*
             *  There is space for this NUL.  The "add_nul_char" is not part of
             *  the "sz" that was added to "next_ix".
             */
            pFMC->buffer[ pFMC->high_water ] = NUL;
    }

    return sz;
}


static seek_pos_t
fmem_seek (void *cookie, fmem_off_t *p_offset, int dir)
{
    fmem_off_t new_pos;
    fmem_cookie_t *pFMC = cookie;

    switch (dir) {
    case SEEK_SET: new_pos = *p_offset;  break;
    case SEEK_CUR: new_pos = pFMC->next_ix  + *p_offset;  break;
    case SEEK_END: new_pos = pFMC->high_water - *p_offset;  break;

#   if SIZEOF_CHARP == SIZEOF_LONG
    /*
     *  This is how we get our IOCTL's.  There is no official way.
     *  We cannot extract our cookie pointer from the FILE* struct
     *  any other way.  :(  Fortunately, we know that "fmem_off_t"-s are
     *  long-s and we know that sizeof(char*) == sizeof(long)  :-)
     */
    case FMEM_IOCTL_SAVE_BUF:
        pFMC->mode &= ~FLAG_BIT(allocated);
        /* FALLTHROUGH */

    case FMEM_IOCTL_BUF_ADDR:
        *(char**)p_offset = pFMC->buffer;
        new_pos = pFMC->next_ix;
        break;
#   endif

    default:
        goto seek_oops;
    }

    if ((signed)new_pos < 0)
        goto seek_oops;

    if (new_pos > pFMC->buf_size) {
        if (fmem_extend( pFMC, new_pos ))
            return -1; /* errno is set */
    }

    pFMC->next_ix = new_pos;
    return new_pos;

 seek_oops:
    errno = EINVAL;
    return -1;
}


static int
fmem_close( void *cookie )
{
    fmem_cookie_t *pFMC = cookie;

    if (pFMC->mode & FLAG_BIT(allocated))
        free( pFMC->buffer );
    free( pFMC );

    return 0;
}

/*=export_func fmem_ioctl
 *
 *  what:  get information about a string stream
 *
 *  arg: + FILE* + fptr  + the string stream
 *  arg: + int   + req   + the requested data
 *  arg: + void* + ptr   + ptr to result area
 *
 *  ret-type:  int
 *  ret-desc:  zero on success
 *
 *  err: non-zero is returned and @code{errno} is set to @code{EINVAL}.
 *
 *  doc:
 *
 *  This routine surreptitiously slips in a special request.
 *  The commands supported are:
 *
 *  @table @code
 *  @item FMEM_IOCTL_BUF_ADDR
 *
 *    Retrieve the address of the buffer.  Future output to the stream might
 *    cause this buffer to be freed and the contents copied to another buffer.
 *    You must ensure that either you have saved the buffer (see
 *    @code{FMEM_IOCTL_SAVE_BUF} below), or do not do any more I/O to it while
 *    you are using this address.
 *
 *    "ptr" must point to a @code{char*} pointer.
 *
 *  @item FMEM_IOCTL_SAVE_BUF
 *
 *    Do not deallocate the buffer on close.  You would likely want to use
 *    this after writing all the output data and just before closing.
 *    Otherwise, the buffer might get relocated.  Once you have specified
 *    this, the current buffer becomes the client program's resposibility to
 *    @code{free()}.  If more I/O operations are performed, a new buffer
 *    @i{may} get allocated.  @code{fmem_close} will free that new buffer
 *    and the user will remain responsible for @code{free()}-ing this buffer.
 *
 *    "ptr" must point to a @code{char*} pointer.
 *
 *  @end table
 *
 *  The third argument is never optional and must be a pointer to where data
 *  are to be retrieved or stored.  It may be NULL if there are no data to
 *  transfer, but both of these functions currently return the address of the
 *  buffer.  This is implemented as a wrapper around @code{fseek(3C)}, so
 *  the "req" argument must not conflict with @code{SEEK_SET}, @code{SEEK_CUR}
 *  or @code{SEEK_END}.
=*/
int
fmem_ioctl( FILE* fp, int req, void* ptr )
{
    if (fseek( fp, (long)ptr, req ) < 0)
        return -1;
    return 0;
}

/*=export_func fmemopen
 *
 *  what:  Open a stream to a string
 *
 *  arg: + void*  + buf  + buffer to use for i/o +
 *  arg: + size_t + len  + size of the buffer +
 *  arg: + char*  + mode + mode string, a la fopen(3C) +
 *
 *  ret-type:  FILE*
 *  ret-desc:  a stdio FILE* pointer
 *
 *  err:  NULL is returned and errno is set to @code{EINVAL} or @code{ENOSPC}.
 *
 *  doc:
 *
 *  This function requires underlying @var{libc} functionality:
 *  either @code{fopencookie(3GNU)} or @code{funopen(3BSD)}.
 *
 *  If @code{buf} is @code{NULL}, then a buffer is allocated.
 *  It is allocated to size @code{len}, unless that is zero.
 *  If @code{len} is zero, then @code{getpagesize()} is used and the buffer
 *  is marked as "extensible".  Any allocated memory is @code{free()}-ed
 *  when @code{fclose(3C)} is called.
 *
 *  The mode string is interpreted as follows.  If the first character of
 *  the mode is:
 *
 *  @table @code
 *  @item a
 *  Then the string is opened in "append" mode.  Append mode is always
 *  extensible.  In binary mode, "appending" will begin from the end of the
 *  initial buffer.  Otherwise, appending will start at the first NUL character
 *  in the initial buffer (or the end of the buffer if there is no NUL
 *  character).
 *
 *  @item w
 *  Then the string is opened in "write" mode.  If a buffer is supplied, then
 *  writing (and reading) will be constrained to the size of that initial
 *  buffer and the buffer will not be reallocated.  If the buffer is not
 *  supplied and the length is specified as zero, then it will be allocated and
 *  reallocated as needed.  If the buffer is not supplied and the length is
 *  non-zero, then a buffer of that size is allocated and the writing is
 *  constrained to that size.
 *
 *  @item r
 *  Then the string is opened in "read" mode.
 *  @end table
 *
 *  @noindent
 *  If it is not one of these three, the open fails and @code{errno} is
 *  set to @code{EINVAL}.  These initial characters may be followed by:
 *
 *  @table @code
 *  @item +
 *  The buffer is marked as extensible and both reading and writing is enabled.
 *
 *  @item b
 *  The I/O is marked as "binary" and a trailing NUL will not be inserted into
 *  the buffer (if it fits).  It will fit if the buffer is extensible (opened
 *  in append mode or write mode without a provided buffer).
 *
 *  @item x
 *  This is ignored.
 *  @end table
 *
 *  @noindent
 *  Any other letters following the inital 'a', 'w' or 'r' will cause an error.
=*/
FILE *
fmemopen(void *buf, size_t len, const char *pMode)
{
    fmem_cookie_t *pFMC;

    {
        mode_bits_t mode;

        if (fmem_getmode(pMode, &mode) != 0) {
            errno = EINVAL;
            return NULL;
        }

        pFMC = malloc(sizeof(fmem_cookie_t));
        if (pFMC == NULL) {
            errno = ENOMEM;
            return NULL;
        }

        pFMC->mode = mode;
    }

    if (buf == NULL)
        pFMC->mode |= FLAG_BIT(allocated);
    pFMC->pg_size = getpagesize();

    if ((pFMC->mode & FLAG_BIT(allocated)) == 0) {
        /*
         *  User allocated buffer.  User responsible for disposal.
         */
        if (pFMC->mode & FLAG_BIT(truncate)) {
            pFMC->next_ix = 0;
        }

        /*
         *  User allocated buffer.  User responsible for disposal.
         */
        else if (pFMC->mode & FLAG_BIT(append)) {
            if (pFMC->mode &FLAG_BIT(binary)) {
                pFMC->next_ix = len;
            } else {
                /*
                 * append text mode -- find the end of the buffer
                 * (the first NUL character)
                 */
                char *p = pFMC->buffer = buf;
                while ((*p != NUL) && (++(pFMC->next_ix) < len))  p++;
            }
        }

        /*
         *  text mode - NUL terminate buffer
         */
        if (   ((pFMC->mode & FLAG_BIT(binary)) == 0)
            && (pFMC->next_ix < len)) {
            pFMC->buffer[pFMC->next_ix] = NUL;
        }
    }

    else if ((pFMC->mode & FLAG_BIT(create)) == 0) {
        /*
         *  No user supplied buffer and we are reading.  Nonsense.
         */
        errno = EINVAL;
        free( pFMC );
        return NULL;
    }

    else {
        /*
         *  Not pre-allocated and we are writing.  We must allocate the buffer.
         *  Whenever we allocate something beyond what is specified (zero, in
         *  this case), the mode had best include "write".
         */
        if (len == 0) {
            if ((pFMC->mode & FLAG_BIT(write)) == 0) {
                errno = EINVAL;
                free( pFMC );
                return NULL;
            }

            len = pFMC->pg_size;
            /*
             * if we allocated it - we can expand it
             */
            pFMC->mode |= FLAG_BIT(append);
        }

        /*
         *  Unallocated file space is set to NULs.  Emulate that.
         */
        pFMC->buffer = calloc(1, len);
        if (pFMC->buffer == NULL) {
            errno = ENOMEM;
            free( pFMC );
            return NULL;
        }

        pFMC->next_ix = 0;
    }

    pFMC->buf_size   = len;
    pFMC->high_water = (pFMC->mode & FLAG_BIT(binary))
                   ? len : strlen(pFMC->buffer);

#ifdef TEST_FMEMOPEN
    saved_cookie = pFMC;
#endif

    {
        cookie_read_function_t* pRd = (pFMC->mode & FLAG_BIT(read))
             ? (cookie_read_function_t*)fmem_read  : NULL;
        cookie_write_function_t* pWr = (pFMC->mode & FLAG_BIT(write))
            ? (cookie_write_function_t*)fmem_write : NULL;
#if defined(HAVE_FOPENCOOKIE)
        cookie_io_functions_t iof;
        iof.read  = pRd;
        iof.write = pWr;
        iof.seek  = (cookie_seek_function_t* )fmem_seek;
        iof.close = (cookie_close_function_t*)fmem_close;

        return fopencookie( pFMC, pMode, iof );
#elif defined(HAVE_FUNOPEN)
        return funopen( pFMC, pRd, pWr,
                        (cookie_seek_function_t* )fmem_seek,
                        (cookie_close_function_t*)fmem_close );
#else
#       include "We have neither fopencookie(3GNU) nor funopen(3BSD)"
#endif
    }
}

#endif /* ENABLE_FMEMOPEN */
/*
 * Local Variables:
 * mode: C
 * c-file-style: "stroustrup"
 * tab-width: 4
 * indent-tabs-mode: nil
 * End:
 * end of agen5/fmemopen.c
*/