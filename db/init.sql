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
    red     integer[500][500] not null,
    gre     integer[500][500] not null,
    blu     integer[500][500] not null,
    hue     integer[500][500] not null,
    sat     integer[500][500] not null,
    val     integer[500][500] not null,
    edge    integer[500][500] not null,
    dilate  integer[500][500] not null,
    erode   integer[500][500] not null,
    label   boolean
);
