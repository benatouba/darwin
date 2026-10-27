! C interoperability only; the scientific calculation is unmodified NCAR RIP.
subroutine cjc_mcape(nx,ny,nz,p,t,q,z,terrain,psfc,cape2,cin2,lcl,lfc,filename,nchar,ierr) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: nx,ny,nz,nchar
  real(c_double),intent(in) :: p(nx,ny,nz),t(nx,ny,nz),q(nx,ny,nz),z(nx,ny,nz)
  real(c_double),intent(in) :: terrain(nx,ny),psfc(nx,ny)
  real(c_double),intent(out) :: cape2(nx,ny),cin2(nx,ny),lcl(nx,ny),lfc(nx,ny)
  character(c_char),intent(in) :: filename(nchar)
  integer(c_int),intent(out) :: ierr
  real(c_double),allocatable :: cape(:,:,:),cin(:,:,:),pf(:,:,:),pn(:,:,:),tn(:,:,:),qn(:,:,:),zn(:,:,:)
  character(len=nchar) :: fname
  character(len=512) :: errmsg
  integer :: i
  allocate(cape(nx,ny,nz),cin(nx,ny,nz),pf(nz,nx,ny),pn(nz,nx,ny),tn(nz,nx,ny),qn(nz,nx,ny),zn(nz,nx,ny))
  do i=1,nchar
    fname(i:i)=filename(i)
  end do
  ierr=0
  errmsg=''
  call dcapecalc2d(p,t,q,z,terrain,psfc,cape,cin,pf,pn,tn,qn,zn, &
    1.0d20,nx,ny,nz,1,fname,ierr,errmsg)
  cape2=cape(:,:,nz)
  cin2=cin(:,:,nz)
  lcl=cin(:,:,nz-1)
  lfc=cin(:,:,nz-2)
  deallocate(cape,cin,pf,pn,tn,qn,zn)
end subroutine cjc_mcape

subroutine cjc_slp(nx,ny,nz,z,t,p,q,slp,ierr) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: nx,ny,nz
  real(c_double),intent(in) :: z(nx,ny,nz),t(nx,ny,nz),p(nx,ny,nz),q(nx,ny,nz)
  real(c_double),intent(out) :: slp(nx,ny)
  integer(c_int),intent(out) :: ierr
  real(c_double),allocatable :: ts(:,:),tsea(:,:)
  integer,allocatable :: level(:,:)
  character(len=512) :: errmsg
  allocate(ts(nx,ny),tsea(nx,ny),level(nx,ny))
  ierr=0
  errmsg=''
  call dcomputeseaprs(nx,ny,nz,z,t,p,q,slp,tsea,ts,level,ierr,errmsg)
  deallocate(ts,tsea,level)
end subroutine cjc_slp
