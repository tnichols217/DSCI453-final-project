create table db_version
(
    version varchar(5) not null
        constraint db_version_pk
            primary key
);

insert into db_version (version) values ('dev');

create table images
(
    id bigserial not null
        constraint images_pk
            primary key,
    R       integer[500][500] not null,
    G       integer[500][500] not null,
    B       integer[500][500] not null,
    H       integer[500][500] not null,
    S       integer[500][500] not null,
    V       integer[500][500] not null,
    edge    integer[500][500],
    dilate  integer[500][500],
    erode   integer[500][500]
);
